import os
import sys
import json
import time
import random
import logging
import argparse
from PIL import Image
from pathlib import Path

from app.utils.logger import Logger

wan_logger = Logger(logger_name="wan2gp", loglevel=logging.ERROR)

# Import from wgp.py
from wgp import (
    generate_video,
    get_model_def,
    get_model_filename,
    get_default_settings,
    get_base_model_type,
    server_config,
    args as wgp_args,
    gen_lock,
)

def get_available_models():
    """Get list of available models from defaults directory."""
    models = {}
    defaults_dir = Path("defaults")
    if not defaults_dir.exists():
        return models
    
    for json_file in defaults_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                config = json.load(f)
                model_info = config.get("model", {})
                model_name = json_file.stem
                models[model_name] = {
                    "name": model_info.get("name", model_name),
                    "architecture": model_info.get("architecture", "unknown"),
                    "description": model_info.get("description", "No description available")
                }
        except (json.JSONDecodeError, KeyError):
            continue
    
    return models

def get_model_capabilities(model_name):
    """Determine model capabilities based on architecture and name."""
    try:
        model_def = get_model_def(model_name)
        if not model_def:
            return {"type": "unknown"}
        
        architecture = model_def.get("architecture", "")
        
        # LTX Video models
        if "ltxv" in architecture:
            return {
                "type": "ltxv",
                "supports_text_to_video": True,
                "supports_image_to_video": True,
                "supports_audio": False,
                "supports_control": True,
                "requires_long_prompts": True,
                "max_frames": 1800,
                "default_fps": 30
            }
        
        # VACE models
        elif "vace" in architecture:
            caps = {
                "type": "vace",
                "supports_text_to_video": True,
                "supports_image_to_video": True,
                "supports_control": True,
                "supports_reference_injection": True,
                "supports_audio": True,
                "max_frames": "257",
                "default_fps": 25
            }
            
            # VACE Multitalk combination
            if "multitalk" in architecture:
                caps.update({
                    "supports_audio": True,
                    "supports_multitalk": True,
                    "requires_audio": True
                })
            else:
                caps["supports_audio"] = False
            
            return caps
        
        # Pure Multitalk models
        elif "multitalk" in architecture or model_name in ["multitalk", "multitalk_720p"]:
            return {
                "type": "multitalk",
                "supports_text_to_video": False,
                "supports_image_to_video": True,
                "supports_audio": True,
                "supports_multitalk": True,
                "requires_audio": True,
                "requires_image": True,
                "max_frames": 257,
                "default_fps": 25
            }
        
        # Default capabilities
        else:
            return {
                "type": "other",
                "supports_text_to_video": True,
                "supports_image_to_video": True,
                "supports_audio": False,
                "max_frames": 257,
                "default_fps": 25
            }
            
    except Exception:
        return {"type": "unknown"}

def create_send_cmd_callback():
    """Create a callback function for progress updates."""
    def send_cmd(command, data=None):
        if command == "progress":
            if isinstance(data, list) and len(data) >= 2:
                if len(data) == 3:
                    step, total_steps, status = data
                    wan_logger.info(f"Progress: {step}/{total_steps} - {status}")
                else:  # [progress_value, status]
                    progress_value, status = data
                    wan_logger.info(f"Status: {status}")
            else:
                wan_logger.info(f"Progress: {data}")
        elif command == "status":
            wan_logger.info(f"Status: {data}")
        elif command == "error":
            wan_logger.error(f"Error: {data}")
        elif command == "output":
            wan_logger.info("Generation step completed")
        else:
            wan_logger.info(f"Command: {command}, Data: {data}")
    
    return send_cmd

def create_state_object(model_type, model_filename):
    """Create a state object for the generation process."""
    return {
        "model_type": model_type,
        "model_filename": model_filename,
        "refresh": 0,
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
            "process_status": "process:main",
            "pause_msg": "",
            "abort": False,
        },
        "loras": [],
        "loras_presets": {},
        "loras_names": [],
        "validate_success": 1,
    }

def create_task_object(task_id, params):
    """Create a task object for the generation queue."""
    return {
        "id": task_id,
        "params": params,
        "metadata": {
            "created_at": time.time(),
            "model_type": params.get("model_type", "unknown"),
        }
    }

def load_and_process_image(image_path):
    """Load and process input image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    return image

def get_audio_prompt_type(args, capabilities):
    """Determine the correct audio prompt type based on arguments and model capabilities."""
    if not capabilities.get("supports_audio", False):
        return ""

    audio1 = getattr(args, 'audio1', None)
    if not audio1:
        return ""

    audio2 = getattr(args, 'audio2', None)
    if not audio2:
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

def get_video_prompt_type(args, capabilities):
    """Determine the correct video prompt type based on arguments and model capabilities."""
    video_prompt_type = ""

    # Handle reference image types for VACE models
    if capabilities.get("type") == "vace":
        ref_type = getattr(args, 'reference_image_type', 'inject-people')
        if ref_type == 'inject-people':
            video_prompt_type += "I"
        elif ref_type == 'inject-landscape':
            video_prompt_type += "KI"
        elif ref_type == 'inject-frames':
            video_prompt_type += "FI"
        # 'none' adds nothing

    return video_prompt_type

def load_reference_images(reference_image_paths):
    """Load reference images from file paths."""
    if not reference_image_paths or len(reference_image_paths) == 0:
        return None

    images = []
    for path in reference_image_paths:
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Reference image not found: {path}")
        image = Image.open(path).convert("RGB")
        images.append(image)

    return images if images else None

def setup_model_configuration(args):
    """Set up model configuration based on the selected model."""
    model_name = args.model
    
    # Get model definition
    model_def = get_model_def(model_name)
    if not model_def:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(get_available_models().keys())}")
    
    # Get model filename
    model_filename = get_model_filename(model_name)
    if not model_filename:
        raise ValueError(f"Could not determine filename for model '{model_name}'")
    
    return model_name, model_filename

def validate_arguments(args, capabilities):
    """Validate arguments based on model capabilities."""
    errors = []

    # Check required arguments based on model type
    if capabilities.get("requires_image", False) and not args.image:
        errors.append("--image is required for this model")

    if capabilities.get("requires_audio", False) and not args.audio1:
        errors.append("--audio1 is required for this model")

    # Check file existence
    if args.image and not os.path.exists(args.image):
        errors.append(f"Image file not found: {args.image}")

    if args.audio1 and not os.path.exists(args.audio1):
        errors.append(f"Audio file not found: {args.audio1}")

    if getattr(args, 'audio2', None) and not os.path.exists(args.audio2):
        errors.append(f"Audio file not found: {args.audio2}")

    # Check reference images for VACE models
    if capabilities.get("type") == "vace":
        ref_type = getattr(args, 'reference_image_type', 'inject-people')
        if ref_type != 'none':
            ref_images = getattr(args, 'reference_images', None)
            if not ref_images:
                errors.append(f"--reference-images is required when --reference-image-type is '{ref_type}'")
            else:
                for ref_path in ref_images:
                    if not os.path.exists(ref_path):
                        errors.append(f"Reference image not found: {ref_path}")

        # Check frame positions if using inject-frames
        if ref_type == 'inject-frames':
            if not getattr(args, 'frame_positions', None):
                errors.append("--frame-positions is required when using --reference-image-type inject-frames")

    # LTX Video specific validations
    if capabilities.get("type") == "ltxv":
        if args.frames < 17:
            errors.append("LTX Video models require at least 17 frames")
        if (args.frames - 17) % 8 != 0:
            errors.append("LTX Video frame count must be 17 + multiple of 8 (e.g., 17, 25, 33, 41, 49, 57, 65, 73, 81...)")

    if errors:
        wan_logger.debug("Validation errors:\n" + "\n".join(f"  - {error}" for error in errors))
        raise ValueError("Validation errors:\n" + "\n".join(f"  - {error}" for error in errors))

def create_generation_parameters(args, image, model_type, capabilities):
    """Create parameters for the generate_video function."""
    # Get model defaults
    defaults = get_default_settings(model_type)

    # Handle seed
    seed = args.seed if args.seed != -1 else random.randint(0, 2**32 - 1)

    # Base parameters that all models need
    params = {
        # Core generation parameters
        "image_mode": 0,  # 0 = video generation, 1 = image generation
        "prompt": args.prompt,
        "negative_prompt": getattr(args, 'negative_prompt', ""),
        "resolution": getattr(args, 'resolution', defaults.get("resolution", "832x480")),
        "video_length": args.frames,
        "batch_size": 1,
        "seed": seed,
        "force_fps": str(getattr(args, 'fps', 0)) if getattr(args, 'fps', 0) != 0 else "",
        "num_inference_steps": int(getattr(args, 'steps', defaults.get("num_inference_steps", 30))),
        "guidance_scale": float(getattr(args, 'guidance', defaults.get("guidance_scale", 3.0))),
        "guidance2_scale": float(defaults.get("guidance2_scale", 5.0)),
        "guidance3_scale": float(defaults.get("guidance3_scale", 5.0)),
        "switch_threshold": float(defaults.get("switch_threshold", 0.5)),
        "switch_threshold2": float(defaults.get("switch_threshold2", 0.3)),
        "guidance_phases": defaults.get("guidance_phases", 1),
        "model_switch_phase": defaults.get("model_switch_phase", 1),
        "audio_guidance_scale": float(defaults.get("audio_guidance_scale", 1.0)) if capabilities.get("supports_audio") else 1.0,
        "flow_shift": float(defaults.get("flow_shift", 7.0)),
        "sample_solver": defaults.get("sample_solver", "euler"),
        "embedded_guidance_scale": float(defaults.get("embedded_guidance_scale", 6.0)),
        "repeat_generation": defaults.get("repeat_generation", 1),
        "multi_prompts_gen_type": defaults.get("multi_prompts_gen_type", 0),
        "multi_images_gen_type": defaults.get("multi_images_gen_type", 0),
        "skip_steps_cache_type": defaults.get("skip_steps_cache_type", ""),
        "skip_steps_multiplier": float(defaults.get("skip_steps_multiplier", 1.5)),
        "skip_steps_start_step_perc": float(defaults.get("skip_steps_start_step_perc", 20.0)),
        "activated_loras": defaults.get("activated_loras", []),
        "loras_multipliers": defaults.get("loras_multipliers", ""),

        # Image parameters
        "image_prompt_type": "S" if image else "",  # "S" for start image if provided
        "image_start": image,
        "image_end": None,
        "model_mode": "",

        # Video parameters
        "video_source": None,
        "keep_frames_video_source": "",
        "video_prompt_type": get_video_prompt_type(args, capabilities),
        "image_refs": load_reference_images(getattr(args, 'reference_images', None)) if capabilities.get("type") == "vace" else None,
        "frames_positions": getattr(args, 'frame_positions', ""),
        "video_guide": None,
        "image_guide": None,
        "keep_frames_video_guide": "",
        "denoising_strength": float(1.0),
        "video_guide_outpainting": "",
        "video_mask": None,
        "image_mask": None,
        "control_net_weight": float(1.0),
        "control_net_weight2": float(1.0),
        "mask_expand": int(0),

        # Audio parameters
        "audio_guide": getattr(args, 'audio1', "") if capabilities.get("supports_audio", True) else "",
        "audio_guide2": getattr(args, 'audio2', None) if capabilities.get("supports_audio") and getattr(args, 'audio2', None) else None,
        "audio_source": getattr(args, 'audio1', "") if capabilities.get("supports_audio") and getattr(args, 'audio1', None) else "",
        "audio_prompt_type": get_audio_prompt_type(args, capabilities),
        "speakers_locations": getattr(args, 'speakers', ""),

        # Sliding window parameters
        "sliding_window_size": int(getattr(args, 'sliding_window_size', 0)),  # Use 0 as default, will be set to 129 if needed
        "sliding_window_overlap": int(0),
        "sliding_window_color_correction_strength": float(1.0),
        "sliding_window_overlap_noise": float(0.0),
        "sliding_window_discard_last_frames": int(getattr(args, 'sliding_window_discard_last_frames', 0)),

        # Reference and injection parameters (VACE specific)
        "image_refs_relative_size": float(1.0),
        "remove_background_images_ref": int(getattr(args, 'remove_background_reference', 1 if capabilities.get("type") == "vace" else 0)),

        # Post-processing parameters
        "temporal_upsampling": "",
        "spatial_upsampling": "",
        "film_grain_intensity": float(0.0),
        "film_grain_saturation": float(0.0),

        # MMAudio parameters
        "MMAudio_setting": int(0),
        "MMAudio_prompt": "",
        "MMAudio_neg_prompt": "",

        # Advanced parameters
        "RIFLEx_setting": int(defaults.get("RIFLEx_setting", 0)),
        "NAG_scale": float(defaults.get("NAG_scale", 1.0)),  # Use 1.0 as default like multitalk
        "NAG_tau": float(defaults.get("NAG_tau", 0.0)),
        "NAG_alpha": float(defaults.get("NAG_alpha", 0.0)),
        "slg_switch": int(defaults.get("slg_switch", 0)),
        "slg_layers": str(defaults.get("slg_layers", "9")),  # Ensure string type
        "slg_start_perc": float(defaults.get("slg_start_perc", 10.0)),
        "slg_end_perc": float(defaults.get("slg_end_perc", 90.0)),
        "apg_switch": int(defaults.get("apg_switch", 0)),
        "cfg_star_switch": int(defaults.get("cfg_star_switch", 0)),
        "cfg_zero_step": int(defaults.get("cfg_zero_step", 0)),  # Use 0 as default like multitalk
        "prompt_enhancer": str(getattr(args, 'prompt_enhancer', "")),
        "min_frames_if_references": int(1),  # Use 1 as default like multitalk
        "override_profile": int(getattr(args, 'memory_profile', -1)),
    }

    # Set sliding window size based on model capabilities and frame count
    if params["sliding_window_size"] == 0:
        if capabilities.get("supports_multitalk") or args.frames > 129:
            params["sliding_window_size"] = int(129)
        else:
            params["sliding_window_size"] = int(args.frames)

    return params

def create_argument_parser():
    """Create argument parser with model-specific arguments."""
    parser = argparse.ArgumentParser(
        description="Universal CLI for video generation using various models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model-specific examples:

LTX Video (Text-to-Video):
  %(prog)s --model ltxv_distilled --prompt "A cat walking in a garden" --frames 81 --output video.mp4

LTX Video (Image-to-Video):
  %(prog)s --model ltxv_13B --image input.jpg --prompt "The person starts walking" --frames 129 --output video.mp4

VACE Multitalk (Audio-driven):
  %(prog)s --model vace_multitalk_14B --image person.jpg --audio1 speech.wav --prompt "Person speaking" --frames 81 --output video.mp4

Pure Multitalk:
  %(prog)s --model multitalk --image person.jpg --audio1 speech.wav --prompt "Person talking" --frames 81 --output video.mp4

VACE ControlNet:
  %(prog)s --model vace_14B --image input.jpg --prompt "Person dancing" --frames 81 --output video.mp4 --reference-images person1.jpg person2.jpg --reference-image-type inject-people

Performance optimization:
  %(prog)s --model ltxv_distilled --prompt "A cat walking" --frames 81 --output video.mp4 --compile --quant bf16 --attention flash --memory-profile 1
        """
    )

    # Core arguments (required for generation, but not for utility commands)
    parser.add_argument("--model", help="Model to use for generation")
    parser.add_argument("--output", help="Output video path (.mp4)")
    parser.add_argument("--frames", type=int, help="Number of frames to generate")
    parser.add_argument("--prompt", help="Text prompt to guide generation")

    # Core optional arguments
    parser.add_argument("--image", help="Input image path (required for image-to-video models)")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt")
    parser.add_argument("--steps", type=int, default=10, help="Number of inference steps (model default if not specified)")
    parser.add_argument("--guidance", type=float, default=3.0, help="CFG guidance scale (model default if not specified)")
    parser.add_argument("--resolution", default="832x480", help="Output resolution WxH (e.g., 832x480)")
    parser.add_argument("--fps", type=int, default=25, help="Override FPS (0 = use model defaults)")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")

    # Audio arguments (for Multitalk models)
    parser.add_argument("--audio1", default="", help="Primary audio file path (required for audio-driven models)")
    parser.add_argument("--audio2", default="", help="Secondary audio file path (for dual-speaker scenarios)")
    parser.add_argument("--speakers", default="", help="Speaker bounding boxes: 'L:R L:R' or 'L:T:R:B L:T:R:B' in percent")
    parser.add_argument("--speaker-mode", choices=['auto', 'parallel', 'sequential'], default='sequential',
                        help="Speaker mode for dual audio: auto=XA (auto separation), parallel=PAB (parallel), sequential=CAB (row/sequential)")
    parser.add_argument("--clean-audio", action='store_true',
                        help="Remove background music/noise for better lip sync (adds 'V' flag to audio_prompt_type)")

    # VACE-specific arguments
    parser.add_argument("--remove-background-reference", type=int, choices=[0, 1], default=0,
                        help="Remove background from reference images (VACE models)")
    parser.add_argument("--reference-image-type", choices=['none', 'inject-people', 'inject-landscape', 'inject-frames'],
                        default='inject-people',
                        help="Reference image injection type for VACE models: none=no injection, inject-people=people/objects only, inject-landscape=landscape then people/objects, inject-frames=positioned frames then people/objects")
    parser.add_argument("--reference-images", nargs='+', help="Reference image file paths (for VACE models)")
    parser.add_argument("--frame-positions", help="Space-separated frame positions for reference injection (e.g., '1 10 L' where L=last frame)")

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
    parser.add_argument("--prompt-enhancer", default="", help="Prompt enhancement method")

    # Utility arguments
    parser.add_argument("--list-models", action='store_true', help="List available models and exit")
    parser.add_argument("--model-info", help="Show detailed information about a specific model")

    return parser

def list_available_models():
    """List all available models with their capabilities."""
    models = get_available_models()
    if not models:
        print("No models found in defaults/ directory")
        return

    print("Available Models:")
    print("=" * 80)

    # Group models by type
    model_groups = {
        "LTX Video": [],
        "VACE": [],
        "Multitalk": [],
        "Other": []
    }

    for model_name, info in models.items():
        capabilities = get_model_capabilities(model_name)
        model_type = capabilities.get("type", "other")

        if model_type == "ltxv":
            model_groups["LTX Video"].append((model_name, info, capabilities))
        elif model_type == "vace":
            model_groups["VACE"].append((model_name, info, capabilities))
        elif model_type == "multitalk":
            model_groups["Multitalk"].append((model_name, info, capabilities))
        else:
            model_groups["Other"].append((model_name, info, capabilities))

    for group_name, group_models in model_groups.items():
        if group_models:
            print(f"\n{group_name} Models:")
            print("-" * 40)
            for model_name, info, capabilities in group_models:
                print(f"  {model_name}")
                print(f"    Name: {info['name']}")
                print(f"    Architecture: {info['architecture']}")
                if capabilities.get("supports_audio"):
                    print(f"    Audio: ✓ (requires --audio1)")
                if capabilities.get("requires_image"):
                    print(f"    Image: ✓ (requires --image)")
                if capabilities.get("supports_control"):
                    print(f"    Control: ✓")
                max_frames = capabilities.get("max_frames", "Unknown")
                print(f"    Max Frames: {max_frames}")
                print()

def show_model_info(model_name):
    """Show detailed information about a specific model."""
    models = get_available_models()
    if model_name not in models:
        wan_logger.error(f"Model '{model_name}' not found.")
        wan_logger.info(f"Available models: {', '.join(models.keys())}")
        return

    info = models[model_name]
    capabilities = get_model_capabilities(model_name)

    print(f"Model Information: {model_name}")
    print("=" * 60)
    print(f"Name: {info['name']}")
    print(f"Architecture: {info['architecture']}")
    print(f"Description: {info['description']}")
    print()

    print("Capabilities:")
    print(f"  Type: {capabilities.get('type', 'unknown')}")
    print(f"  Text-to-Video: {'✓' if capabilities.get('supports_text_to_video') else '✗'}")
    print(f"  Image-to-Video: {'✓' if capabilities.get('supports_image_to_video') else '✗'}")
    print(f"  Audio Support: {'✓' if capabilities.get('supports_audio') else '✗'}")
    print(f"  Control Support: {'✓' if capabilities.get('supports_control') else '✗'}")
    print(f"  Max Frames: {capabilities.get('max_frames', 'Unknown')}")
    print(f"  Default FPS: {capabilities.get('default_fps', 'Unknown')}")
    print()

    print("Required Arguments:")
    required_args = ["--model", "--output", "--frames", "--prompt"]
    if capabilities.get("requires_image"):
        required_args.append("--image")
    if capabilities.get("requires_audio"):
        required_args.append("--audio1")

    for arg in required_args:
        print(f"  {arg}")

    print()
    print("Example Usage:")
    example_cmd = f"python generate.py --model {model_name} --output video.mp4 --frames 81 --prompt \"Your prompt here\""
    if capabilities.get("requires_image"):
        example_cmd += " --image input.jpg"
    if capabilities.get("requires_audio"):
        example_cmd += " --audio1 speech.wav"
    print(f"  {example_cmd}")

def apply_performance_settings(args):
    """Apply performance settings to server configuration."""
    from wgp import server_config

    # Apply transformer compilation setting
    if getattr(args, 'compile', False):
        server_config["compile"] = 1
        wan_logger.info("Transformer compilation enabled")

    # Apply quantization setting
    if hasattr(args, 'quant') and args.quant:
        server_config["transformer_quantization"] = args.quant
        wan_logger.info(f"Model quantization set to: {args.quant}")

    # Apply dtype policy
    if hasattr(args, 'dtype') and args.dtype:
        server_config["transformer_dtype_policy"] = args.dtype
        wan_logger.info(f"Transformer dtype policy set to: {args.dtype}")

    # Apply attention mechanism
    if hasattr(args, 'attention') and args.attention != "auto":
        server_config["attention_mode"] = args.attention
        wan_logger.info(f"Attention mechanism set to: {args.attention}")

    # Apply VRAM settings
    if hasattr(args, 'vram_safety') and args.vram_safety != 0.8:
        server_config["vram_safety_coefficient"] = args.vram_safety
        wan_logger.info(f"VRAM safety coefficient set to: {args.vram_safety}")

    if hasattr(args, 'reserved_mem') and args.reserved_mem != 0.95:
        server_config["max_percentage_reserved_memory"] = args.reserved_mem
        wan_logger.info(f"Max reserved memory percentage set to: {args.reserved_mem}")

    # Apply preload setting
    if hasattr(args, 'preload') and args.preload > 0:
        server_config["preload_model_policy"] = [args.preload]
        wan_logger.info(f"Model preload set to: {args.preload} MB")

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
        wan_logger.error(f"Could not find generated video with seed {seed} in {outputs_dir}")
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
        wan_logger.error(f"Error moving file from {generated_file} to {desired_output_path}: {e}")
        return None

def generate_video_cli(task, send_cmd, state, model_type, model_filename, **params):
    """
    CLI-compatible wrapper for generate_video that properly handles sliding window generation.

    This function implements the task processing logic that the web UI handles through
    the process_tasks generator pattern, ensuring that all sliding windows are processed
    when num_frames > sliding_window_size.
    """
    from shared.utils.thread_utils import AsyncStream, async_run

    # Create async stream for communication
    com_stream = AsyncStream()
    internal_send_cmd = com_stream.output_queue.push

    def generate_video_error_handler():
        try:
            generate_video(
                task=task,
                send_cmd=internal_send_cmd,
                state=state,
                model_type=model_type,
                model_filename=model_filename,
                mode="",
                **params
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc().split('\n')[:-1]
            print('\n'.join(tb))
            internal_send_cmd("error", str(e))
        finally:
            internal_send_cmd("exit", None)

    # Start the generation process asynchronously
    async_run(generate_video_error_handler)

    # Process commands from the generation function
    while True:
        cmd, data = com_stream.output_queue.next()

        if cmd == "exit":
            break
        elif cmd == "error":
            raise Exception(data)
        elif cmd == "output":
            # This is called after each window is completed
            # For CLI, we just pass it through to the original send_cmd
            send_cmd("output")
        elif cmd == "progress":
            send_cmd("progress", data)
        elif cmd == "status":
            send_cmd("status", data)
        elif cmd == "info":
            wan_logger.info(data)
        elif cmd == "preview":
            # CLI doesn't need preview handling, just ignore
            pass
        else:
            wan_logger.warning(f"Unknown command received: {cmd}")

    wan_logger.info("Video generation completed successfully")

def main():
    """Main function."""
    parser = create_argument_parser()

    # Parse known args first to handle utility commands
    args, _ = parser.parse_known_args()
    # Handle utility commands first (don't require other arguments)
    if args.list_models:
        list_available_models()
        return 0

    if args.model_info:
        show_model_info(args.model_info)
        return 0

    # Now parse all arguments for generation
    args = parser.parse_args()

    # Check required arguments for generation
    required_args = ['model', 'output', 'frames', 'prompt']
    missing_args = [arg for arg in required_args if not getattr(args, arg)]
    if missing_args:
        parser.error(f"the following arguments are required: {', '.join('--' + arg for arg in missing_args)}")

    try:
        wan_logger.debug(f"Model: {args.model}")
        wan_logger.debug(f"Output: {args.output}")
        wan_logger.debug(f"Frames: {args.frames}")

        # Get model capabilities
        capabilities = get_model_capabilities(args.model)
        wan_logger.debug(f"Model Type: {capabilities.get('type', 'unknown')}")

        # Validate arguments
        validate_arguments(args, capabilities)

        # Load image if provided
        image = None
        if args.image:
            wan_logger.debug(f"Loading image: {args.image}")
            image = load_and_process_image(args.image)
            wan_logger.debug(f"Image loaded: {image.size}")
        elif capabilities.get("requires_image"):
            raise ValueError("This model requires an input image (--image)")

        # Set up model configuration
        #print("Setting up model configuration...")
        model_type, model_filename = setup_model_configuration(args)
        wan_logger.debug(f"Model type: {model_type}")
        wan_logger.debug(f"Model filename: {model_filename}")

        # Apply performance settings
        apply_performance_settings(args)

        # Create state object
        wan_logger.debug("Creating state object...")
        state = create_state_object(model_type, model_filename)

        # Create generation parameters
        wan_logger.debug("Creating generation parameters...")
        params = create_generation_parameters(args, image, model_type, capabilities)

        # Create task object
        task_id = int(time.time() * 1000)
        task = create_task_object(task_id, params)

        # Create send_cmd callback
        send_cmd = create_send_cmd_callback()

        # Create output directory if needed
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        wan_logger.info("\nStarting video generation")
        wan_logger.info("This may take several minutes depending on your hardware...")

        # Debug audio parameters if audio model
        if capabilities.get("supports_audio"):
            wan_logger.debug("Audio parameters:")
            wan_logger.debug(f"  - audio_guide: {params['audio_guide']}")
            wan_logger.debug(f"  - audio_guide2: {params['audio_guide2']}")
            wan_logger.debug(f"  - audio_prompt_type: {params['audio_prompt_type']}")
            wan_logger.debug(f"  - clean_audio flag: {'V' in params['audio_prompt_type']}")

        # Debug VACE parameters if VACE model
        if capabilities.get("type") == "vace":
            wan_logger.debug("VACE parameters:")
            wan_logger.debug(f"  - video_prompt_type: {params['video_prompt_type']}")
            wan_logger.debug(f"  - reference_image_type: {getattr(args, 'reference_image_type', 'inject-people')}")
            wan_logger.debug(f"  - reference_images: {len(params['image_refs']) if params['image_refs'] else 0} images")
            wan_logger.debug(f"  - frame_positions: {params['frames_positions']}")
            wan_logger.debug(f"  - remove_background: {params['remove_background_images_ref']}")

        # Debug performance parameters
        wan_logger.debug("Performance parameters:")
        wan_logger.debug(f"  - memory_profile: {getattr(args, 'memory_profile', -1)}")
        wan_logger.debug(f"  - compile: {getattr(args, 'compile', False)}")
        wan_logger.debug(f"  - quantization: {getattr(args, 'quant', 'int8')}")
        wan_logger.debug(f"  - dtype: {getattr(args, 'dtype', 'fp16')}")
        wan_logger.debug(f"  - attention: {getattr(args, 'attention', 'auto')}")
        # Store the desired output path for later use
        desired_output_path = os.path.abspath(args.output)
        wan_logger.debug(f"Wan2GP video output path: {desired_output_path}")

        # Use CLI-compatible wrapper that properly handles sliding window generation
        generate_video_cli(
            task=task,
            send_cmd=send_cmd,
            state=state,
            model_type=model_type,
            model_filename=model_filename,
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
        )

        try:
            generated_file = find_and_move_generated_video(desired_output_path, params['seed'], params['prompt'])
            if generated_file:
                wan_logger.info(f"Video saved to: {generated_file}")

                # Check if the output file has audio (for audio models)
                if capabilities.get("supports_audio") and args.audio1:
                    try:
                        import librosa
                        y, sr = librosa.load(generated_file, sr=None)
                        if len(y) > 0:
                            wan_logger.info(f"Output video contains audio (duration: {len(y)/sr:.2f}s, sample rate: {sr}Hz)")
                        else:
                            wan_logger.warning("Output video appears to have no audio")
                    except Exception as audio_check_error:
                        wan_logger.warning(f"Could not check audio in output video: {audio_check_error}")
            else:
                wan_logger.warning("Generated video found in outputs folder, but could not be moved to specified location")
                wan_logger.warning("Check the outputs folder for your generated video")
        except Exception as move_error:
            wan_logger.warning(f"Video generated successfully but could not be moved: {move_error}")
            wan_logger.warning("Check the outputs folder for your generated video")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
