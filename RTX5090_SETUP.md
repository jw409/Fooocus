# RTX 5090 Setup Guide for Fooocus

This guide provides step-by-step instructions to set up Fooocus with NVIDIA RTX 5090 support.

## Prerequisites

- **Linux** operating system (Ubuntu 22.04+ recommended)
- **Python 3.12+** installed
- **NVIDIA RTX 5090** GPU
- **NVIDIA Driver 570.00+** installed
- **CUDA 12.8** toolkit (will be installed with PyTorch)

## Quick Setup

### 1. Run the automated setup script:

```bash
./setup_rtx5090.sh
```

This script will:
- Install UV package manager
- Create a Python 3.12 virtual environment
- Install PyTorch 2.9.0 with CUDA 12.8 support
- Build xformers from source for SM 120 support
- Install all dependencies
- Create an optimized launch script

### 2. Launch Fooocus:

```bash
./launch_rtx5090.sh
```

## Manual Setup

If you prefer to set up manually:

### 1. Create virtual environment:

```bash
python3.12 -m venv venv_rtx5090
source venv_rtx5090/bin/activate
```

### 2. Install PyTorch with CUDA 12.8:

```bash
pip install torch==2.9.0.dev20250726+cu128 torchvision==0.21.0.dev20250726+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 3. Build xformers from source:

```bash
python build_xformers.py
```

### 4. Install remaining dependencies:

```bash
pip install -r requirements_rtx5090.txt
```

### 5. Launch with RTX 5090 optimizations:

```bash
export USE_RTX5090=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python launch.py --xformers --opt-sdp-attention --no-half-vae
```

## Verify Installation

Run the GPU test script to verify everything is working:

```bash
python test_gpu.py
```

Expected output:
- PyTorch 2.9.0 with CUDA 12.8
- RTX 5090 detected with SM 120 compute capability
- ~102 TFLOPS FP16 performance
- xformers working with memory efficient attention

## Optimizations Applied

The RTX 5090 setup includes several optimizations:

1. **PyTorch 2.9.0 nightly** - Latest version with RTX 5090 support
2. **CUDA 12.8** - Required for SM 120 compute capability
3. **xformers custom build** - Memory efficient attention for SM 120
4. **--xformers** - Enable memory efficient attention
5. **--opt-sdp-attention** - Use PyTorch's optimized scaled dot product attention
6. **--no-half-vae** - Avoid half precision VAE for better stability
7. **expandable_segments** - Better GPU memory allocation

## Troubleshooting

### CUDA not available
- Ensure NVIDIA driver 570.00+ is installed: `nvidia-smi`
- Verify CUDA 12.8 compatible PyTorch is installed

### xformers build fails
- Ensure you have build tools: `sudo apt install build-essential`
- Check CUDA toolkit: `nvcc --version`
- Try manual build with verbose output: `pip install -v xformers`

### Out of memory errors
- The RTX 5090 has 32GB VRAM, ensure no other processes are using GPU
- Use `nvidia-smi` to check GPU memory usage
- Try reducing batch size in Fooocus settings

### Performance issues
- Verify GPU is running at full speed: `nvidia-smi -q -d PERFORMANCE`
- Check power management: `sudo nvidia-smi -pm 1`
- Ensure PCIe Gen 5 x16 connection

## Notes

- This setup is compatible with all RTX 50-series GPUs (5080, 5070 Ti, 5070)
- The custom xformers build is required for SM 120 architecture support
- Python 3.12+ is required for optimal performance with modern async features