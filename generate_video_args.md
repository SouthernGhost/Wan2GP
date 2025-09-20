# generate_video Function Arguments Documentation

This document describes all arguments for the `generate_video` function in `wgp.py`, including their types, purposes, and possible values.

## Core Parameters

**task** (dict) - **REQUIRED** - Task object containing generation parameters and metadata
  - Keys:
    - "id" (int) - Unique task identifier
    - "params" (dict) - **REQUIRED** - Contains all generation parameters (see Params Dictionary section below)
    - "repeats" (int) - Number of times to repeat generation
    - "length" (int) - Video length in frames
    - "steps" (int) - Number of inference steps
    - "prompt" (str) - Text prompt for generation
    - "start_image_labels" (list) - Labels for start images
    - "end_image_labels" (list) - Labels for end images
    - "start_image_data" (PIL.Image objects or None) - Start image data
    - "end_image_data" (PIL.Image objects or None) - End image data
    - "start_image_data_base64" (list or None) - Base64 encoded start images
    - "end_image_data_base64" (list or None) - Base64 encoded end images

**send_cmd** (function) - **REQUIRED** - Callback function for sending progress updates and status messages
  - Accepts: command string ("progress", "output", "error", "exit") and optional data

**state** (dict) - **REQUIRED** - Application state object containing configuration and runtime data
  - Keys:
    - "model_filename" (str) - Current model filename
    - "model_type" (str) - Current model type
    - "advanced" (bool) - Advanced UI mode flag
    - "last_model_per_family" (dict) - Last used model per family
    - "last_resolution_per_group" (dict) - Last used resolution per group
    - "gen" (dict) - Generation state containing queue and progress info
    - "loras" (list) - Available LoRA models
    - "loras_presets" (dict) - LoRA preset configurations
    - "loras_names" (list) - LoRA model names
    - "validate_success" (int) - Validation status flag

**model_type** (str) - Model type identifier
  - Values: "multitalk", "i2v", "t2v", "t2v_1.3B", "fun_inp_1.3B", "vace_1.3B", "ltxv_13B", "recam_1.3B", "phantom_1.3B", "phantom_14B", "hunyuan_custom", "hunyuan_custom_audio", "hunyuan_custom_edit", "fantasy", etc.

**model_filename** (str) - Filename of the model to use for generation

**mode** (str) - Generation mode
  - Values: normal generation (empty string) or editing modes starting with "edit_" (e.g., "edit_remux", "edit_postprocessing")

## Params Dictionary Structure

The **params** dictionary within the **task** argument contains all the generation parameters. It includes all the individual arguments listed below, plus additional metadata:

**Required Keys in params:**
- All individual function arguments (image_mode, prompt, negative_prompt, etc.) - see sections below
- "state" (dict) - Reference to the application state object
- "model_type" (str) - Model type identifier
- "model_filename" (str) - Model filename

**Optional Keys in params:**
- "mode" (str) - Generation mode (empty string for normal generation)
- Image/video file paths as strings for media inputs
- PIL.Image objects for image inputs
- Lists of images for multi-image inputs

**Argument Optionality:**
- **REQUIRED**: task, send_cmd, state (via params), model_type (via params), model_filename (via params)
- **OPTIONAL**: All other arguments have default values and can be omitted

## Image and Video Generation

**image_mode** (int) - **OPTIONAL** - Image generation mode
  - Values: 0 (video output), >0 (image output)
  - Default: 0

**prompt** (str) - **OPTIONAL** - Text prompt for generation, can contain multiple lines separated by newlines
  - Default: empty string or model-specific default

**negative_prompt** (str) - **OPTIONAL** - Negative text prompt to avoid certain features
  - Default: empty string

**resolution** (str) - **OPTIONAL** - Output resolution in "WIDTHxHEIGHT" format
  - Common values: "1280x720", "832x480", "1024x1024", "512x512"
  - Default: model-specific (e.g., "1280x720" for 720p models, "832x480" for others)

**video_length** (int) - **OPTIONAL** - Number of frames to generate
  - Range: typically 1-200+ frames depending on model
  - Default: 81

**batch_size** (int) - **OPTIONAL** - Number of videos to generate in parallel
  - Range: typically 1-4
  - Default: 1

**seed** (int) - **OPTIONAL** - Random seed for reproducible generation
  - Values: -1 (random), 0-999999999 (specific seed)
  - Default: -1

**force_fps** (int) - **OPTIONAL** - Override FPS value
  - Values: 0 (use model defaults), or specific FPS like 24, 30, 60
  - Default: 0

## Inference Parameters

**num_inference_steps** (int) - **OPTIONAL** - Number of denoising steps during generation
  - Range: typically 10-100, minimum 20 for some models like ltxv_13B
  - Default: 30

**guidance_scale** (float) - **OPTIONAL** - Primary classifier-free guidance scale
  - Range: typically 1.0-20.0
  - Default: 5.0

**guidance2_scale** (float) - **OPTIONAL** - Secondary guidance scale for multi-phase generation
  - Range: typically 1.0-20.0
  - Default: model-specific

**guidance3_scale** (float) - **OPTIONAL** - Tertiary guidance scale for three-phase generation
  - Range: typically 1.0-20.0
  - Default: model-specific

**switch_threshold** (float) - **OPTIONAL** - Noise level threshold for switching between guidance phases
  - Range: 0.0-1.0
  - Default: model-specific

**switch_threshold2** (float) - **OPTIONAL** - Second threshold for three-phase guidance
  - Range: 0.0-1.0
  - Default: model-specific

**guidance_phases** (int) - **OPTIONAL** - Number of guidance phases
  - Values: 1, 2, or 3 (depends on model support)
  - Default: 1

**model_switch_phase** (int) - **OPTIONAL** - Phase at which to switch models if applicable
  - Values: 1, 2, or 3
  - Default: model-specific

**audio_guidance_scale** (float) - **OPTIONAL** - Guidance scale for audio-conditioned generation
  - Range: 1.0-20.0
  - Default: 4.0 (for audio-capable models)

**flow_shift** (float) - **OPTIONAL** - Flow shift parameter for motion control
  - Range: 1.0-25.0
  - Default: 7.0 for non-720p models, 5.0 for others

**sample_solver** (str) - **OPTIONAL** - Sampling solver method
  - Values: depends on model definition, can be "euler", "dpm", "unipc", "dpm++", "causvid", etc.
  - Default: "unipc" for most models, "euler" for multitalk

**embedded_guidance_scale** (float) - **OPTIONAL** - Scale for embedded guidance features
  - Range: 1.0-20.0
  - Default: 6.0

## Generation Control

**repeat_generation** (int) - **OPTIONAL** - Number of times to repeat the generation process
  - Range: 1-25
  - Default: 1

**multi_prompts_gen_type** (int) - **OPTIONAL** - How to handle multiple prompts
  - Values: 0 (separate videos), 1 (single video)
  - Default: 0

**multi_images_gen_type** (int) - **OPTIONAL** - How to handle multiple input images
  - Values: 0 (separate processing), 1 (combined processing)
  - Default: 0

**skip_steps_cache_type** (str) - **OPTIONAL** - Type of step skipping cache to use
  - Values: "" (none), "tea" (TeaCache), "mag" (MagCache)
  - Default: "" (disabled)

**skip_steps_multiplier** (float) - **OPTIONAL** - Multiplier for skip steps optimization
  - Values: 1.5 (x1.5 speedup), 1.75 (x1.75 speedup), 2.0 (x2 speedup), 2.25 (x2.25 speedup)
  - Default: 1.5

**skip_steps_start_step_perc** (float) - **OPTIONAL** - Percentage of steps before starting skip optimization
  - Range: 0-100
  - Default: 20

## LoRA and Model Customization

**activated_loras** (list) - **OPTIONAL** - List of activated LoRA model names
  - Default: [] (empty list)

**loras_multipliers** (str) - **OPTIONAL** - String specifying LoRA strength multipliers
  - Format: space or newline separated values, lines starting with # are ignored
  - Default: "" (empty string, uses 1.0 for all LoRAs)

## Image Input Parameters

**image_prompt_type** (str) - **OPTIONAL** - String of letters indicating image input types
  - Letters: "S" (start image), "E" (end image), "I" (reference images), "V" (video source), "L" (last video)
  - Examples: "S", "SE", "SVI", "L"
  - Default: "" (empty string) or "S" for i2v models

**image_start** (PIL.Image.Image or list) - **OPTIONAL** - Starting image(s) for generation
  - Can be single PIL.Image or list of PIL.Image objects
  - Default: None

**image_end** (PIL.Image.Image or list) - **OPTIONAL** - Ending image(s) for generation
  - Can be single PIL.Image or list of PIL.Image objects
  - Default: None

**image_refs** (list) - **OPTIONAL** - List of reference images for style/character consistency
  - List of PIL.Image objects
  - Default: None

**image_guide** (PIL.Image.Image) - **OPTIONAL** - Control image for image-to-video generation
  - Single PIL.Image object
  - Default: None

**image_mask** (PIL.Image.Image) - **OPTIONAL** - Mask image for inpainting operations
  - Single PIL.Image object
  - Default: None

## Video Input Parameters

**video_source** (str) - Path to source video file for video-to-video generation

**keep_frames_video_source** (str) - Number of frames to keep from source video
  - Values: "" (all frames), "1" (first frame), "a:b" (range), space-separated values

**video_prompt_type** (str) - String of letters indicating video processing types
  - Letters: "V" (control video), "G" (guide), "U" (unchanged), "P" (pose), "D" (depth), "E" (edges), "S" (shapes), "L" (flow), "C" (recolorize), "M" (inpainting), "I" (reference images)
  - Examples: "", "UV", "PV", "DV", "EV", "SV", "LV", "CV", "MV", "V", "PDV", "PSV", "PLV", "DSV", "DLV", "SLV"

**frames_positions** (str) - Space-separated frame positions for reference injection

**video_guide** (str) - Path to control video file

**keep_frames_video_guide** (str) - Frame selection pattern for control video
  - Values: "" (all), "1" (first), "a:b" (range), "-1" (last), space-separated values

**denoising_strength** (float) - Strength of denoising
  - Range: 0.0-1.0, where 1.0 = full generation, 0.0 = no denoising

**video_guide_outpainting** (str) - Outpainting dimensions as "left top right bottom" string

## Mask and Control Parameters

**video_mask** (str) - Path to video mask file for inpainting
**control_net_weight** (float) - Weight for primary ControlNet conditioning
**control_net_weight2** (float) - Weight for secondary ControlNet conditioning
**mask_expand** (int) - Number of pixels to expand masks

## Audio Parameters

**audio_guide** (str) - Path to primary audio file for audio-driven generation

**audio_guide2** (str) - Path to secondary audio file for dual-speaker scenarios

**audio_source** (str) - Path to custom audio soundtrack file

**audio_prompt_type** (str) - String of letters indicating audio processing types
  - Letters: "A" (audio source), "B" (audio source #2), "V" (vocals), "X" (extended audio features)
  - Examples: "", "A", "B", "AB", "AV", "BX"

**speakers_locations** (str) - Bounding box coordinates for speaker locations
  - Format: space-separated bounding box coordinates for multiple speakers

## Sliding Window Parameters

**sliding_window_size** (int) - Size of each sliding window in frames
**sliding_window_overlap** (int) - Number of overlapping frames between windows
**sliding_window_color_correction_strength** (float) - Strength of color correction between windows
**sliding_window_overlap_noise** (float) - Noise level for window transitions
**sliding_window_discard_last_frames** (int) - Number of frames to discard from each window end

## Reference Image Parameters

**image_refs_relative_size** (float) - Relative size scaling for reference images
**remove_background_images_ref** (int) - Whether to remove background from reference images (0=no, 1=yes)

## Post-processing Parameters

**temporal_upsampling** (str) - Temporal upsampling method name
  - Values: "" (disabled), "rife2" (Rife x2 frames/s), "rife4" (Rife x4 frames/s)

**spatial_upsampling** (str) - Spatial upsampling method name
  - Values: "" (disabled), "lanczos1.5" (Lanczos x1.5), "lanczos2" (Lanczos x2.0)

**film_grain_intensity** (float) - Intensity of film grain effect
  - Range: 0.0-1.0 (0.0 = disabled)

**film_grain_saturation** (float) - Saturation of film grain effect
  - Range: 0.0-1.0

## Audio Generation Parameters

**MMAudio_setting** (int) - MMAudio generation mode
  - Values: 0 (off), 1 (on - generate soundtrack based on video)

**MMAudio_prompt** (str) - Text prompt for MMAudio soundtrack generation
  - Format: 1-2 keywords describing desired audio

**MMAudio_neg_prompt** (str) - Negative prompt for MMAudio generation
  - Format: 1-2 keywords to avoid in audio

## Advanced Parameters

**RIFLEx_setting** (int) - RIFLEx enhancement setting
  - Values: 0 (off), 1 (on)

**NAG_scale** (float) - Negative Augmented Generation scale
  - Range: typically 1.0-3.0

**NAG_tau** (float) - NAG tau parameter
  - Range: typically 0.0-1.0

**NAG_alpha** (float) - NAG alpha parameter
  - Range: typically 0.0-1.0

**slg_switch** (int) - Semantic Latent Guidance switch
  - Values: 0 (off), 1 (on)

**slg_layers** (str) - Comma-separated list of SLG layer indices
  - Format: comma-separated integers (e.g., "9", "8,9,10")

**slg_start_perc** (float) - SLG start percentage
  - Range: 0.0-100.0

**slg_end_perc** (float) - SLG end percentage
  - Range: 0.0-100.0

**apg_switch** (int) - Adaptive Progressive Guidance switch
  - Values: 0 (off), 1 (on)

**cfg_star_switch** (int) - CFG Star guidance switch
  - Values: 0 (off), 1 (on)

**cfg_zero_step** (int) - CFG zero step parameter
  - Range: typically 0-50

## Miscellaneous Parameters

**model_mode** (str) - Specific model mode or variant
  - Values: depends on model definition, can be empty string or model-specific modes

**prompt_enhancer** (str) - Prompt enhancement method
  - Values: depends on server configuration and enhancer settings

**min_frames_if_references** (int) - Minimum frames when using reference images
  - Range: typically 1-100

**override_profile** (int) - **OPTIONAL** - Memory profile override
  - Values: -1 (default), 0-5 (specific memory profiles: LowRAM_LowVRAM, etc.)
  - Default: -1

## Summary of Required vs Optional Arguments

### REQUIRED Arguments (must be provided):
1. **task** - Task dictionary with params containing all generation parameters
2. **send_cmd** - Callback function for progress updates
3. **state** (via task.params) - Application state dictionary
4. **model_type** (via task.params) - Model type identifier
5. **model_filename** (via task.params) - Model filename

### OPTIONAL Arguments (have defaults):
All other arguments are optional and have sensible defaults. The most commonly used optional arguments include:
- **prompt** - Text prompt (defaults to empty or model-specific)
- **resolution** - Output resolution (defaults to model-specific)
- **video_length** - Number of frames (defaults to 81)
- **num_inference_steps** - Denoising steps (defaults to 30)
- **seed** - Random seed (defaults to -1 for random)
- **guidance_scale** - CFG scale (defaults to 5.0)

### Dictionary Arguments Deep Dive:

**task.params Dictionary** contains:
- All individual function arguments as key-value pairs
- "state" key pointing to the state dictionary
- "model_type" and "model_filename" keys
- Media file paths as strings or PIL.Image objects
- Optional "mode" key for editing operations

**state Dictionary** contains:
- Model configuration ("model_filename", "model_type", "advanced")
- UI state ("last_model_per_family", "last_resolution_per_group")
- Generation queue and progress ("gen" subdictionary)
- LoRA information ("loras", "loras_presets", "loras_names")
- Validation flags ("validate_success")
