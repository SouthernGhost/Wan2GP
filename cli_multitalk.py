import os
import sys
import argparse
import json
import torch
import glob
from PIL import Image

# Reuse repo utilities
from wgp import (
    get_model_def,
    get_model_fps,
    get_model_min_frames_and_step,
    get_base_model_type,
    load_models,
    release_model,
    server_config,
    args as wgp_args,
)
from models.wan.multitalk.multitalk import (
    get_full_audio_embeddings,
    get_window_audio_embeddings,
    parse_speakers_locations,
)
from shared.utils.audio_video import save_video, combine_video_with_audio_tracks


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def parse_resolution(res: str) -> tuple[int, int]:
    try:
        w, h = res.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise ValueError(f"Invalid resolution '{res}'. Expected WxH like 960x544.")


def main():

    parser = argparse.ArgumentParser("Multitalk CLI - image + one/two audio → talking video")
    parser.add_argument("--image", required=True, help="Path to input image (1-2 persons)")
    parser.add_argument("--audio1", required=True, help="Audio file for person 1 (wav/mp3/mp4)")
    parser.add_argument("--audio2", default=None, help="Optional audio file for person 2")
    parser.add_argument("--output", required=True, help="Output video path (.mp4)")
    parser.add_argument("--prompt", default="", help="Text prompt to guide generation")
    parser.add_argument("--neg-prompt", default="", help="Negative prompt")
    parser.add_argument("--frames", type=int, default=81, help="Frames to generate (rounded to model step)")
    parser.add_argument("--steps", type=int, default=40, help="Denoising steps")
    parser.add_argument("--guidance", type=float, default=5.0, help="CFG guidance scale")
    parser.add_argument("--resolution", default="480x720", help="Output resolution WxH (empty = use variant default)")
    parser.add_argument("--variant", default="480p", choices=["480p","720p"], help="Multitalk variant; default 480p")
    parser.add_argument("--fps", type=int, default=0, help="Override FPS (0 = use model defaults)")
    parser.add_argument("--seed", type=int, default=-1, help="Seed (-1 random)")
    parser.add_argument("--quant", default="int8", help="Model quantization preference (int8/bf16/fp16)")
    parser.add_argument("--dtype", default="fp16", help="Transformer dtype policy: '', fp16 or bf16")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--memory-profile", type=int, default=-1, choices=[-1, 1, 2, 3, 4, 5],
                        help="Memory profile: -1=default, 1=HighRAM_HighVRAM, 2=HighRAM_LowVRAM, 3=LowRAM_HighVRAM, 4=LowRAM_LowVRAM, 5=VeryLowRAM_LowVRAM")
    parser.add_argument("--compile", action="store_true", help="Enable transformer compilation for faster inference")
    parser.add_argument("--attention", default="auto", choices=["auto", "sdpa", "sage", "sage2", "flash", "xformers"],
                        help="Attention mechanism: auto, sdpa, sage, sage2, flash, xformers")
    parser.add_argument("--preload", type=int, default=0, help="Megabytes of model to preload in VRAM (0=auto)")
    parser.add_argument("--vram-safety", type=float, default=0.8, help="VRAM safety coefficient (0.1-1.0)")
    parser.add_argument("--reserved-mem", type=float, default=0.95, help="Max percentage of reserved memory (0.1-1.0)")
    parser.add_argument("--speakers", default="", help="Optional speakers bboxes: 'L:R L:R' or 'L:T:R:B L:T:R:B' in percent")
    parser.add_argument("--audio-combine", default="auto", choices=["auto","add","para"], help="Two-speaker audio combine mode")

    args = parser.parse_args()

    image = load_image(args.image)

    # Select model variant and default resolution
    model_type = "multitalk" if args.variant == "480p" else "multitalk_720p"
    default_res = "960x544" if args.variant == "480p" else "1280x720"
    res_str = args.resolution if len(args.resolution) > 0 else default_res
    width, height = parse_resolution(res_str)
    model_def = get_model_def(model_type)
    if model_def is None:
        raise RuntimeError("Model definition for 'multitalk' not found. Ensure defaults/multitalk.json exists.")

    base_model_type = get_base_model_type(model_type)
    fps = args.fps if args.fps > 0 else get_model_fps(base_model_type)

    # Normalize frames to model's latent step
    frames_minimum, frames_step = get_model_min_frames_and_step(model_type)
    frames = max(args.frames, frames_minimum)
    frames = (frames // frames_step) * frames_step + 1

    # Configure wgp.py settings for advanced memory management
    print(f"Configuring memory management...")
    print(f"Memory profile: {args.memory_profile}")
    print(f"Attention mode: {args.attention}")
    print(f"VRAM preload: {args.preload}MB")
    print(f"VRAM safety coefficient: {args.vram_safety}")
    print(f"Reserved memory max: {args.reserved_mem}")

    # Update wgp.py's global settings to match CLI arguments
    wgp_args.preload = str(args.preload)
    wgp_args.vram_safety_coefficient = args.vram_safety
    wgp_args.perc_reserved_mem_max = args.reserved_mem
    wgp_args.attention = args.attention
    wgp_args.compile = args.compile

    # Update server config for quantization and dtype
    server_config["transformer_quantization"] = args.quant
    server_config["transformer_dtype_policy"] = args.dtype
    server_config["attention_mode"] = args.attention
    if args.compile:
        server_config["compile"] = "transformer"

    # Use wgp.py's advanced model loading with memory management
    print(f"Loading model with advanced memory management...")
    wan_model, offloadobj = load_models(model_type, override_profile=args.memory_profile)
    print(f"Model loaded successfully with profile {args.memory_profile}")
    # offloadobj handles automatic memory management - no direct usage needed

    # Prepare multitalk audio embeddings
    combination = ("add" if args.audio_combine=="add" else "para" if args.audio_combine=="para" else ("para" if args.audio2 else "add"))
    full_audio_embs, mixed_audio = get_full_audio_embeddings(
        audio_guide1=args.audio1,
        audio_guide2=args.audio2,
        combination_type=combination,
        num_frames=frames,
        fps=fps,
        sr=16000,
        padded_frames_for_embeddings=0,
        min_audio_duration=frames/float(fps),
    )

    # Speakers placement (optional)
    speakers_bboxes = None
    if args.speakers:
        speakers_bboxes, err = parse_speakers_locations(args.speakers)
        if err:
            raise ValueError(err)

    # Build inputs for a single window generation (no sliding window in CLI)
    audio_proj_split = get_window_audio_embeddings(full_audio_embs, audio_start_idx=0, clip_length=frames)

    # Run generation
    torch.set_grad_enabled(False)
    out = wan_model.generate(
        input_prompt=args.prompt,
        image_start=image,
        input_frames=None,
        input_ref_images=None,
        input_video=None,
        denoising_strength=1.0,
        prefix_frames_count=0,
        frame_num=frames,
        batch_size=1,
        height=height,
        width=width,
        fit_into_canvas=True,
        shift=7,
        sample_solver="euler",
        sampling_steps=args.steps,
        guide_scale=args.guidance,
    n_prompt=getattr(args, "neg_prompt", ""),
        seed=(None if args.seed == -1 else args.seed),
        enable_RIFLEx=False,
        VAE_tile_size=0,
        joint_pass=False,
        slg_layers=None,
        slg_start=0.0,
        slg_end=1.0,
        audio_proj=audio_proj_split,
        audio_context_lens=None,
        audio_scale=None,
        model_type=model_type,
        model_mode=None,
        speakers_bboxes=speakers_bboxes,
        color_correction_strength=1,
        image_mode=0,
        window_no=0,
        set_header_text=None,
        pre_video_frame=None,
        video_prompt_type="",
        original_input_ref_images=[],
    )

    samples = out["samples"] if isinstance(out, dict) and "samples" in out else out
    if samples is None:
        raise RuntimeError("Generation failed: no samples returned")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tmp_video = os.path.splitext(args.output)[0] + "_raw.mp4"
    save_video(tensor=samples[None], save_file=tmp_video, fps=fps, nrow=1, normalize=True, value_range=(-1, 1))

    # Write mixed audio to temp file if present and mux
    if mixed_audio is not None and len(mixed_audio) > 0:
        import soundfile as sf
        tmp_audio = os.path.splitext(args.output)[0] + "_audio.wav"
        sf.write(tmp_audio, mixed_audio, 16000)
        combine_video_with_audio_tracks(tmp_video, [tmp_audio], args.output)
        try:
            os.remove(tmp_audio)
        except Exception:
            pass
        try:
            os.remove(tmp_video)
        except Exception:
            pass
    else:
        # No audio provided/kept
        os.replace(tmp_video, args.output)

    print(f"Saved: {args.output}")

    # Clean up memory
    print("Cleaning up memory...")
    release_model()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleanup complete.")


if __name__ == "__main__":
    main()


