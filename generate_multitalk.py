#!/usr/bin/env python3
"""
Proper CLI script for Multitalk video generation using the generate_video function from wgp.py.
This script avoids the attribute errors of cli_multitalk.py by properly using the Wan2GP pipeline.
"""

import os
import sys
import random
import time
import argparse
from PIL import Image
from transformers.utils import logging
logging.set_verbosity_error

# Import from wgp.py
from wgp import (
    generate_video,
    get_model_def,
    get_model_filename,
    get_default_settings,
    server_config,
    args as wgp_args
)

os.environ["GRADIO_LANG"] = "en"

def create_send_cmd_callback():
    """Create a callback function for progress updates and status messages."""
    def send_cmd(command, data=None):
        if command == "progress":
            if isinstance(data, list) and len(data) >= 2:
                if len(data) == 3:  # [step, total_steps, status]
                    step, total_steps, status = data
                    print(f"Progress: {step}/{total_steps} - {status}")
                else:  # [progress_value, status]
                    progress_value, status = data
                    print(f"Status: {status}")
            else:
                print(f"Progress: {data}")
        elif command == "error":
            print(f"Error: {data}")
        elif command == "output":
            print("Output generated")
        elif command == "preview":
            # Preview data - just acknowledge
            pass
        elif command == "exit":
            print("Generation completed")
        else:
            print(f"Command: {command}, Data: {data}")
    
    return send_cmd

def create_state_object(model_type, model_filename):
    """Create a proper state object required by generate_video."""
    state = {
        "model_filename": model_filename,
        "model_type": model_type,
        "advanced": False,
        "last_model_per_family": {},
        "last_resolution_per_group": {},
        "gen": {
            "queue": [],
            "in_progress": True,
            "progress_status": "",
            "progress_phase": ("", -1),
            "num_inference_steps": 30,
            "prompt_no": 1,
            "prompts_max": 1,
            "total_generation": 1,
            "repeat_no": 0,
            "total_windows": 1,
            "window_no": 1,
            "extra_orders": 0,
            "extra_windows": 0,
            "file_list": [],
            "file_settings_list": [],
            "header_text": "",
            "refresh": int(time.time() * 1000),
            "process_status": "process:main",  # Initialize process_status to prevent None error
            "pause_msg": "",
            "abort": False,
        },
        "loras": [],
        "loras_presets": {},
        "loras_names": [],
        "validate_success": 1,
    }
    return state

def create_task_object(task_id, params):
    """Create a proper task object required by generate_video."""
    task = {
        "id": task_id,
        "params": params.copy(),
        "repeats": params.get("repeat_generation", 1),
        "length": params.get("video_length", 81),
        "steps": params.get("num_inference_steps", 30),
        "prompt": params.get("prompt", ""),
        "start_image_labels": [],
        "end_image_labels": [],
        "start_image_data": None,
        "end_image_data": None,
        "start_image_data_base64": None,
        "end_image_data_base64": None,
    }
    return task

def load_and_process_image(image_path):
    """Load and process input image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    return image

def get_audio_prompt_type(args):
    """Determine the correct audio prompt type based on arguments."""
    if not getattr(args, 'audio2', None):
        # Single speaker
        base_type = "A"
    else:
        # Dual speakers - map speaker mode to audio prompt type
        speaker_mode = getattr(args, 'speaker_mode', 'parallel')
        if speaker_mode == 'auto':
            base_type = "XA"  # Auto separation
        elif speaker_mode == 'sequential':
            base_type = "CAB"  # Sequential/Row
        else:  # parallel (default)
            base_type = "PAB"  # Parallel

    # Add vocal cleaning flag if requested
    if getattr(args, 'clean_audio', False):
        base_type += "V"

    return base_type



def setup_model_configuration(args):
    """Set up model configuration based on CLI arguments."""
    # Force model type to multitalk (not configurable)
    model_type = "multitalk"
    
    # Get model definition
    model_def = get_model_def(model_type)
    if model_def is None:
        raise RuntimeError("Model definition for 'multitalk' not found. Ensure defaults/multitalk.json exists.")
    
    # Get model filename
    model_filename = get_model_filename(model_type, 
                                      quantization=args.quant, 
                                      dtype_policy=args.dtype)
    
    # Update server config based on CLI arguments
    if hasattr(args, 'memory_profile') and args.memory_profile != -1:
        server_config["profile"] = args.memory_profile
    
    server_config["transformer_quantization"] = args.quant
    server_config["transformer_dtype_policy"] = args.dtype
    server_config["attention_mode"] = args.attention
    
    if args.compile:
        server_config["compile"] = "transformer"
    
    # Update wgp.py global settings
    if hasattr(args, 'preload'):
        wgp_args.preload = str(args.preload)
    if hasattr(args, 'vram_safety'):
        wgp_args.vram_safety_coefficient = args.vram_safety
    if hasattr(args, 'reserved_mem'):
        wgp_args.perc_reserved_mem_max = args.reserved_mem
    wgp_args.attention = args.attention
    wgp_args.compile = args.compile
    
    return model_type, model_filename

def create_generation_parameters(args, image, model_type):
    """Create all parameters needed for generate_video function."""
    # Get default settings for the model
    defaults = get_default_settings(model_type)
    
    # Parse resolution
    if hasattr(args, 'resolution') and args.resolution:
        try:
            width, height = map(int, args.resolution.split('x'))
        except ValueError:
            raise ValueError(f"Invalid resolution format: {args.resolution}. Expected WIDTHxHEIGHT")
    else:
        # Use default resolution from model
        default_res = defaults.get("resolution", "960x544")
        width, height = map(int, default_res.split('x'))
    
    # Handle seed
    seed = args.seed if args.seed != -1 else random.randint(0, 2**32 - 1)
    
    # Create parameters dictionary
    params = {
        # Core generation parameters
        "image_mode": 0,  # Video output
        "prompt": args.prompt,
        "negative_prompt": getattr(args, 'negative_prompt', ''),
        "resolution": f"{width}x{height}",
        "video_length": args.frames,
        "batch_size": 1,
        "seed": seed,
        "force_fps": str(getattr(args, 'fps', 0)) if getattr(args, 'fps', 0) != 0 else "",
        
        # Inference parameters
        "num_inference_steps": int(getattr(args, 'steps', 30)),
        "guidance_scale": float(getattr(args, 'guidance', 5.0)),
        "guidance2_scale": float(defaults.get("guidance2_scale", 5.0)),
        "guidance3_scale": float(defaults.get("guidance3_scale", 5.0)),
        "switch_threshold": float(defaults.get("switch_threshold", 0.5)),
        "switch_threshold2": float(defaults.get("switch_threshold2", 0.3)),
        "guidance_phases": int(defaults.get("guidance_phases", 1)),
        "model_switch_phase": int(defaults.get("model_switch_phase", 1)),
        "audio_guidance_scale": float(defaults.get("audio_guidance_scale", 4.0)),
        "flow_shift": float(defaults.get("flow_shift", 7.0)),
        "sample_solver": str(defaults.get("sample_solver", "euler")),
        "embedded_guidance_scale": float(defaults.get("embedded_guidance_scale", 6.0)),
        
        # Generation control
        "repeat_generation": int(1),
        "multi_prompts_gen_type": int(0),
        "multi_images_gen_type": int(0),
        "skip_steps_cache_type": str(""),
        "skip_steps_multiplier": float(1.5),
        "skip_steps_start_step_perc": float(20.0),
        
        # LoRA parameters
        "activated_loras": [],
        "loras_multipliers": "",
        
        # Image input parameters
        "image_prompt_type": "S",  # Start image
        "image_start": image,
        "image_end": None,
        "model_mode": "",
        
        # Video parameters (not used for multitalk)
        "video_source": None,  # Use None instead of empty string
        "keep_frames_video_source": "",
        "video_prompt_type": "",
        "image_refs": None,
        "frames_positions": "",
        "video_guide": None,  # Use None instead of empty string
        "image_guide": None,
        "keep_frames_video_guide": "",
        "denoising_strength": 1.0,
        "video_guide_outpainting": "",
        "video_mask": None,  # Use None instead of empty string
        "image_mask": None,
        "control_net_weight": 1.0,
        "control_net_weight2": 1.0,
        "mask_expand": 0,
        
        # Audio parameters (will be set up separately)
        "audio_guide": args.audio1 if args.audio1 else "",
        "audio_guide2": getattr(args, 'audio2', None) if getattr(args, 'audio2', None) else None,
        "audio_source": args.audio1,  # Set audio_source to ensure audio combination is triggered
        "audio_prompt_type": get_audio_prompt_type(args),
        "speakers_locations": getattr(args, 'speakers', ""),
        
        # Sliding window parameters
        "sliding_window_size": getattr(args, 'sliding_window_size', 0),
        "sliding_window_overlap": 0,
        "sliding_window_color_correction_strength": 1.0,
        "sliding_window_overlap_noise": 0.0,
        "sliding_window_discard_last_frames": getattr(args, 'sliding_window_discard_last_frames', 0),
        
        # Reference image parameters
        "image_refs_relative_size": 1.0,
        "remove_background_images_ref": 0,
        
        # Post-processing parameters
        "temporal_upsampling": "",
        "spatial_upsampling": "",
        "film_grain_intensity": 0.0,
        "film_grain_saturation": 0.0,
        
        # Audio generation parameters
        "MMAudio_setting": 0,
        "MMAudio_prompt": "",
        "MMAudio_neg_prompt": "",
        
        # Advanced parameters
        "RIFLEx_setting": 0,
        "NAG_scale": 1.0,
        "NAG_tau": 0.0,
        "NAG_alpha": 0.0,
        "slg_switch": 0,
        "slg_layers": "9",  # Default SLG layers as string
        "slg_start_perc": 10.0,
        "slg_end_perc": 90.0,
        "apg_switch": 0,
        "cfg_star_switch": 0,
        "cfg_zero_step": 0,
        
        # Miscellaneous parameters
        "prompt_enhancer": "",
        "min_frames_if_references": 1,
        "override_profile": getattr(args, 'memory_profile', -1),
    }
    
    return params

def find_and_move_generated_video(desired_output_path, seed, prompt):
    """Find the generated video file and move it to the desired location."""
    import glob
    import shutil
    from shared.utils.utils import truncate_for_filesystem
    from wgp import sanitize_file_name

    # Get the outputs directory (same as save_path in wgp.py)
    outputs_dir = os.path.join(os.getcwd(), "outputs")

    # Generate the expected filename pattern (same logic as in wgp.py lines 5442-5445)
    if os.name == 'nt':
        truncated_prompt = sanitize_file_name(truncate_for_filesystem(prompt, 50)).strip()
    else:
        truncated_prompt = sanitize_file_name(truncate_for_filesystem(prompt, 100)).strip()

    # Look for files with the seed and prompt pattern
    pattern = f"*_seed{seed}_{truncated_prompt}*.mp4"
    search_pattern = os.path.join(outputs_dir, pattern)

    # Find matching files
    matching_files = glob.glob(search_pattern)

    if not matching_files:
        # Fallback: look for any recent .mp4 files with the seed
        pattern = f"*_seed{seed}_*.mp4"
        search_pattern = os.path.join(outputs_dir, pattern)
        matching_files = glob.glob(search_pattern)

    if not matching_files:
        print(f"Could not find generated video with seed {seed} in {outputs_dir}")
        return None

    # Get the most recent file (in case there are multiple matches)
    generated_file = max(matching_files, key=os.path.getctime)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(desired_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Move the file to the desired location
    try:
        shutil.move(generated_file, desired_output_path)
        return desired_output_path
    except Exception as e:
        print(f"Error moving file from {generated_file} to {desired_output_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Proper Multitalk CLI - Generate talking videos using the Wan2GP pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --image person.jpg --audio1 speech.wav --output result.mp4 --frames 81 --prompt "A person talking"
  %(prog)s --image couple.jpg --audio1 person1.wav --audio2 person2.wav --output conversation.mp4 --frames 121 --prompt "Two people having a conversation"
        """
    )
    
    # Required arguments
    parser.add_argument("--image", required=True, help="Path to input image (1-2 persons)")
    parser.add_argument("--audio1", required=True, help="Audio file for person 1 (wav/mp3/mp4)")
    parser.add_argument("--output", required=True, help="Output video path (.mp4)")
    parser.add_argument("--frames", type=int, required=True, help="Number of frames to generate")
    parser.add_argument("--prompt", required=True, help="Text prompt to guide generation")
    
    # Optional arguments
    parser.add_argument("--audio2", help="Optional audio file for person 2")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=5.0, help="CFG guidance scale")
    parser.add_argument("--resolution", help="Output resolution WxH (e.g., 960x544)")
    parser.add_argument("--fps", type=int, default=0, help="Override FPS (0 = use model defaults)")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--speakers", default="", help="Speaker bounding boxes: 'L:R L:R' or 'L:T:R:B L:T:R:B' in percent")
    parser.add_argument("--speaker-mode", choices=['auto', 'parallel', 'sequential'], default='parallel',
                        help="Speaker mode for dual audio: auto=XA (auto separation), parallel=PAB (parallel), sequential=CAB (row/sequential)")
    parser.add_argument("--clean-audio", action='store_true', default='False',
                        help="Remove background music/noise for better lip sync (adds 'V' flag to audio_prompt_type)")

    # Sliding window arguments
    parser.add_argument("--sliding-window-size", type=int, default=129, help="Size of each sliding window in frames")
    parser.add_argument("--sliding-window-discard-last-frames", type=int, default=0, help="Number of frames to discard from each window end")
    
    # Performance arguments
    parser.add_argument("--memory-profile", type=int, default=-1, choices=[-1, 1, 2, 3, 4, 5],
                        help="Memory profile: -1=default, 1=HighRAM_HighVRAM, 2=HighRAM_LowVRAM, 3=LowRAM_HighVRAM, 4=LowRAM_LowVRAM, 5=VeryLowRAM_LowVRAM")
    parser.add_argument("--compile", action="store_true", help="Enable transformer compilation for faster inference")
    parser.add_argument("--quant", default="int8", choices=["int8", "bf16", "fp16"], help="Model quantization")
    parser.add_argument("--dtype", default="fp16", choices=["", "fp16", "bf16"], help="Transformer dtype policy")
    parser.add_argument("--attention", default="auto", choices=["auto", "sdpa", "sage", "sage2", "flash", "xformers"],
                        help="Attention mechanism")
    parser.add_argument("--preload", type=int, default=0, help="Megabytes of model to preload in VRAM (0=auto)")
    parser.add_argument("--vram-safety", type=float, default=0.8, help="VRAM safety coefficient (0.1-1.0)")
    parser.add_argument("--reserved-mem", type=float, default=0.95, help="Max percentage of reserved memory (0.1-1.0)")
    
    args = parser.parse_args()
    
    try:
        print("=== Proper Multitalk CLI ===")
        print(f"Input image: {args.image}")
        print(f"Audio 1: {args.audio1}")
        if args.audio2:
            print(f"Audio 2: {args.audio2}")
        print(f"Output: {args.output}")
        print(f"Frames: {args.frames}")
        print(f"Prompt: {args.prompt}")
        print()

        # Validate input files
        print("Validating input files...")
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image file not found: {args.image}")
        if not os.path.exists(args.audio1):
            raise FileNotFoundError(f"Audio file not found: {args.audio1}")
        if getattr(args, 'audio2', None) and not os.path.exists(args.audio2):
            raise FileNotFoundError(f"Audio file not found: {args.audio2}")

        # Check audio file durations
        import librosa
        audio1_duration = librosa.get_duration(path=args.audio1)
        print(f"Audio1 duration: {audio1_duration:.2f} seconds")
        if getattr(args, 'audio2', None):
            audio2_duration = librosa.get_duration(path=args.audio2)
            print(f"Audio2 duration: {audio2_duration:.2f} seconds")

        # Calculate expected video duration - but first check what FPS Multitalk actually uses
        print(f"Checking Multitalk model FPS...")

        # Get model definition to check FPS
        model_def = get_model_def("multitalk")
        if model_def:
            print(f"Model definition found: {model_def.get('name', 'Unknown')}")

        # For now, assume 25 FPS but warn about potential issues
        expected_fps = 25.0
        expected_duration = args.frames / expected_fps
        print(f"Expected video duration: {expected_duration:.2f} seconds ({args.frames} frames at {expected_fps} FPS)")

        if audio1_duration < expected_duration:
            print(f"Warning: Audio1 duration ({audio1_duration:.2f}s) is shorter than expected video duration ({expected_duration:.2f}s)")

        # Check if audio is too short for minimum requirements
        min_duration = 1.0  # Minimum 1 second
        if audio1_duration < min_duration:
            raise ValueError(f"Audio1 duration ({audio1_duration:.2f}s) is too short. Minimum duration is {min_duration}s")

        # Warn if frame count might cause issues with audio processing
        if args.frames < 25:
            print(f"Warning: Frame count ({args.frames}) is quite low. Multitalk works best with 25+ frames.")

        # Check if frame count is compatible with audio duration
        max_frames_for_audio = int(audio1_duration * expected_fps)
        if args.frames > max_frames_for_audio:
            print(f"Warning: Requested frames ({args.frames}) exceeds what audio duration supports ({max_frames_for_audio} frames)")
            print(f"Consider reducing frames to {max_frames_for_audio} or using longer audio")

        print("Input files validated successfully")

        # Load and process input image
        print("Loading input image...")
        image = load_and_process_image(args.image)
        print(f"Image loaded: {image.size}")

        # Set up model configuration
        print("Setting up model configuration...")
        model_type, model_filename = setup_model_configuration(args)
        print(f"Model type: {model_type}")
        print(f"Model filename: {model_filename}")

        # Create state object
        print("Creating state object...")
        state = create_state_object(model_type, model_filename)
        print("State object created successfully")

        # Create generation parameters
        print("Creating generation parameters...")
        params = create_generation_parameters(args, image, model_type)
        params["state"] = state
        params["model_type"] = model_type
        params["model_filename"] = model_filename
        params["mode"] = ""
        print("Generation parameters created successfully")

        # Create task object
        task_id = int(time.time() * 1000)  # Use timestamp as task ID
        task = create_task_object(task_id, params)
        print(f"Task object created with ID: {task_id}")

        # Create send_cmd callback
        send_cmd = create_send_cmd_callback()
        print("Send command callback created")

        # Create output directory if needed
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        print("\n=== Starting video generation ===")
        print("This may take several minutes depending on your hardware...")
        print("Calling generate_video function...")
        print(f"Audio parameters being passed:")
        print(f"  - audio_guide: {params['audio_guide']}")
        print(f"  - audio_guide2: {params['audio_guide2']}")
        print(f"  - audio_prompt_type: {params['audio_prompt_type']}")
        print(f"  - speakers_locations: {params['speakers_locations']}")
        print(f"  - clean_audio flag: {'V' in params['audio_prompt_type']}")

        # Store the desired output path for later use
        desired_output_path = os.path.abspath(args.output)
        print(f"Desired output path: {desired_output_path}")

        # Call generate_video function with correct parameter names
        generate_video(
            task=task,
            send_cmd=send_cmd,
            image_mode=params["image_mode"],
            prompt=params["prompt"],
            negative_prompt=params["negative_prompt"],
            resolution=params["resolution"],
            video_length=params["video_length"],
            batch_size=params["batch_size"],
            seed=params["seed"],
            force_fps=params["force_fps"],
            num_inference_steps=params["num_inference_steps"],
            guidance_scale=params["guidance_scale"],
            guidance2_scale=params["guidance2_scale"],
            guidance3_scale=params["guidance3_scale"],
            switch_threshold=params["switch_threshold"],
            switch_threshold2=params["switch_threshold2"],
            guidance_phases=params["guidance_phases"],
            model_switch_phase=params["model_switch_phase"],
            audio_guidance_scale=params["audio_guidance_scale"],
            flow_shift=params["flow_shift"],
            sample_solver=params["sample_solver"],
            embedded_guidance_scale=params["embedded_guidance_scale"],
            repeat_generation=params["repeat_generation"],
            multi_prompts_gen_type=params["multi_prompts_gen_type"],
            multi_images_gen_type=params["multi_images_gen_type"],
            skip_steps_cache_type=params["skip_steps_cache_type"],
            skip_steps_multiplier=params["skip_steps_multiplier"],
            skip_steps_start_step_perc=params["skip_steps_start_step_perc"],
            activated_loras=params["activated_loras"],
            loras_multipliers=params["loras_multipliers"],
            image_prompt_type=params["image_prompt_type"],
            image_start=params["image_start"],
            image_end=params["image_end"],
            model_mode=params["model_mode"],
            video_source=params["video_source"],
            keep_frames_video_source=params["keep_frames_video_source"],
            video_prompt_type=params["video_prompt_type"],
            image_refs=params["image_refs"],
            frames_positions=params["frames_positions"],
            video_guide=params["video_guide"],
            image_guide=params["image_guide"],
            keep_frames_video_guide=params["keep_frames_video_guide"],
            denoising_strength=params["denoising_strength"],
            video_guide_outpainting=params["video_guide_outpainting"],
            video_mask=params["video_mask"],
            image_mask=params["image_mask"],
            control_net_weight=params["control_net_weight"],
            control_net_weight2=params["control_net_weight2"],
            mask_expand=params["mask_expand"],
            audio_guide=params["audio_guide"],
            audio_guide2=params["audio_guide2"],
            audio_source=params["audio_source"],
            audio_prompt_type=params["audio_prompt_type"],
            speakers_locations=params["speakers_locations"],
            sliding_window_size=params["sliding_window_size"],
            sliding_window_overlap=params["sliding_window_overlap"],
            sliding_window_color_correction_strength=params["sliding_window_color_correction_strength"],
            sliding_window_overlap_noise=params["sliding_window_overlap_noise"],
            sliding_window_discard_last_frames=params["sliding_window_discard_last_frames"],
            image_refs_relative_size=params["image_refs_relative_size"],
            remove_background_images_ref=params["remove_background_images_ref"],
            temporal_upsampling=params["temporal_upsampling"],
            spatial_upsampling=params["spatial_upsampling"],
            film_grain_intensity=params["film_grain_intensity"],
            film_grain_saturation=params["film_grain_saturation"],
            MMAudio_setting=params["MMAudio_setting"],
            MMAudio_prompt=params["MMAudio_prompt"],
            MMAudio_neg_prompt=params["MMAudio_neg_prompt"],
            RIFLEx_setting=params["RIFLEx_setting"],
            NAG_scale=params["NAG_scale"],
            NAG_tau=params["NAG_tau"],
            NAG_alpha=params["NAG_alpha"],
            slg_switch=params["slg_switch"],
            slg_layers=params["slg_layers"],
            slg_start_perc=params["slg_start_perc"],
            slg_end_perc=params["slg_end_perc"],
            apg_switch=params["apg_switch"],
            cfg_star_switch=params["cfg_star_switch"],
            cfg_zero_step=params["cfg_zero_step"],
            prompt_enhancer=params["prompt_enhancer"],
            min_frames_if_references=params["min_frames_if_references"],
            override_profile=params["override_profile"],
            state=params["state"],
            model_type=params["model_type"],
            model_filename=params["model_filename"],
            mode=params["mode"],
        )

        print(f"\n=== Video generation completed successfully! ===")

        # Find the generated video file and move it to the desired location
        try:
            generated_file = find_and_move_generated_video(desired_output_path, params['seed'], params['prompt'])
            if generated_file:
                print(f"✅ Video saved to: {generated_file}")

                # Check if the output file has audio
                try:
                    import librosa
                    y, sr = librosa.load(generated_file, sr=None)
                    if len(y) > 0:
                        print(f"✓ Output video contains audio (duration: {len(y)/sr:.2f}s, sample rate: {sr}Hz)")
                    else:
                        print("⚠ Output video appears to have no audio")
                except Exception as audio_check_error:
                    print(f"⚠ Could not check audio in output video: {audio_check_error}")
            else:
                print("⚠️  Generated video found in outputs folder, but could not be moved to specified location")
                print("Check the outputs folder for your generated video")
        except Exception as move_error:
            print(f"⚠️  Video generated successfully but could not be moved: {move_error}")
            print("Check the outputs folder for your generated video")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
