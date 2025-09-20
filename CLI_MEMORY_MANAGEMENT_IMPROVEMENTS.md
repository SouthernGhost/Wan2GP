# CLI Memory Management Improvements

## Overview

The `cli_multitalk.py` script has been significantly enhanced to use the advanced memory management and optimization methods from `wgp.py`. This provides much better performance, memory efficiency, and hardware compatibility.

## Key Improvements Made

### 1. Advanced Memory Profile Support

**Before:**
- Only supported profiles 1 and 2
- Limited to basic memory configurations

**After:**
- Full support for all 5 memory profiles:
  - Profile 1: HighRAM_HighVRAM (64GB RAM + 24GB VRAM) - Fastest for RTX 3090/4090
  - Profile 2: HighRAM_LowVRAM (64GB RAM + 12GB VRAM) - Versatile for RTX 3070/3080/4070/4080
  - Profile 3: LowRAM_HighVRAM (32GB RAM + 24GB VRAM) - RTX 3090/4090 with limited RAM
  - Profile 4: LowRAM_LowVRAM (32GB RAM + 12GB VRAM) - Recommended for most users
  - Profile 5: VeryLowRAM_LowVRAM (24GB RAM + 10GB VRAM) - Fail-safe for low-end hardware

### 2. Sophisticated Memory Management

**New Features Added:**
- **Attention Mechanism Selection**: `--attention {auto,sdpa,sage,sage2,flash,xformers}`
- **VRAM Preloading**: `--preload MEGABYTES` (0=auto)
- **VRAM Safety Coefficient**: `--vram-safety 0.8` (0.1-1.0)
- **Reserved Memory Control**: `--reserved-mem 0.95` (0.1-1.0)
- **Automatic Memory Cleanup**: Proper model release and garbage collection

### 3. Integration with wgp.py's Advanced Systems

**Before:**
```python
# Manual model instantiation
wan_model = WanAny2V(
    config=cfg,
    checkpoint_dir="ckpts",
    model_filename=model_filename,
    # ... basic parameters only
)
```

**After:**
```python
# Advanced memory management integration
wgp_args.preload = str(args.preload)
wgp_args.vram_safety_coefficient = args.vram_safety
wgp_args.perc_reserved_mem_max = args.reserved_mem
wgp_args.attention = args.attention

# Use wgp.py's sophisticated model loading
wan_model, offloadobj = load_models(model_type, override_profile=args.memory_profile)
```

### 4. Automatic Memory Optimization

The script now uses:
- **offload.profile()**: Advanced memory management with automatic CPU/GPU offloading
- **Budget-based memory allocation**: Different memory budgets per profile
- **Pinned memory optimization**: For profiles 3 and 4
- **Automatic model quantization**: Based on available memory
- **LoRA memory management**: Efficient handling of model adaptations

## New Command Line Options

### Memory Management
```bash
--memory-profile {-1,1,2,3,4,5}    # Memory profile selection
--attention {auto,sdpa,sage,sage2,flash,xformers}  # Attention mechanism
--preload MEGABYTES                # VRAM preloading amount
--vram-safety 0.8                  # VRAM safety coefficient
--reserved-mem 0.95                # Max reserved memory percentage
```

### Usage Examples

**High-end hardware (RTX 4090, 64GB RAM):**
```bash
python cli_multitalk.py --image face.jpg --audio1 speech.wav --output video.mp4 \
    --memory-profile 1 --attention sage2 --preload 1000 --compile
```

**Mid-range hardware (RTX 4070, 32GB RAM):**
```bash
python cli_multitalk.py --image face.jpg --audio1 speech.wav --output video.mp4 \
    --memory-profile 4 --attention sdpa --preload 500
```

**Low-end hardware (RTX 3060, 16GB RAM):**
```bash
python cli_multitalk.py --image face.jpg --audio1 speech.wav --output video.mp4 \
    --memory-profile 5 --attention auto --vram-safety 0.6 --reserved-mem 0.8
```

## Performance Benefits

1. **Better Memory Utilization**: Automatic CPU/GPU memory management
2. **Hardware Optimization**: Profile-specific optimizations for different GPU/RAM combinations
3. **Reduced OOM Errors**: Smart memory budgeting and safety coefficients
4. **Faster Inference**: Optimized attention mechanisms and compilation options
5. **Automatic Cleanup**: Proper memory release after generation

## Technical Implementation

### Memory Profile Configuration
Each profile configures different memory budgets:
- **Profile 1-2**: High VRAM budgets for transformers and text encoders
- **Profile 3**: 70% memory budget with pinned memory
- **Profile 4**: Balanced budgets with pinned memory for dual transformers
- **Profile 5**: Minimal budgets for very low memory systems

### Attention Mechanism Optimization
- **auto**: Automatically selects best available attention
- **sdpa**: PyTorch's scaled dot-product attention (good baseline)
- **sage/sage2**: Optimized attention implementations
- **flash**: FlashAttention for memory efficiency
- **xformers**: Memory-efficient attention from Meta

### Integration Points
The CLI now properly integrates with wgp.py's:
- Global configuration system
- Server config management
- Advanced model loading pipeline
- Memory profiling and optimization
- Automatic model release and cleanup

## Backward Compatibility

All existing CLI arguments remain unchanged. New memory management options are optional with sensible defaults, ensuring existing scripts continue to work while providing access to advanced features when needed.
