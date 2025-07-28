# Fooocus RTX 5090 Fork

This is a fork of [Fooocus](https://github.com/lllyasviel/Fooocus) with specific support for NVIDIA RTX 5090 GPUs.

## What's Different in This Fork

### RTX 5090 Support
- **PyTorch 2.9** with CUDA 12.8 support (required for RTX 5090's compute capability 12.0)
- **Custom xformers build** from source for SM 12.0 architecture
- **Manual dependency management** - automatic dependency installation has been disabled
- **Virtual environment enforcement** - won't run outside of a venv

### New Scripts
- `install_fooocus_rtx5090.sh` - Sets up the complete RTX 5090 environment
- `launch_fooocus_rtx5090.sh` - Launches Fooocus with RTX 5090 optimizations
- `test_rtx5090_setup.py` - Comprehensive test suite for validating setup

## Installation for RTX 5090

1. **Prerequisites**
   - NVIDIA RTX 5090 GPU
   - CUDA 12.8+ installed
   - UV package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`

2. **Install**
   ```bash
   git clone https://github.com/jw409/Fooocus.git
   cd Fooocus
   ./install_fooocus_rtx5090.sh
   ```

3. **Launch**
   ```bash
   ./launch_fooocus_rtx5090.sh
   ```

   To skip pre-flight checks (not recommended):
   ```bash
   SKIP_CHECKS=1 ./launch_fooocus_rtx5090.sh
   ```

## Important Changes

- **No automatic dependency updates** - Dependencies are managed manually to prevent version conflicts
- **Virtual environment required** - The software will refuse to run outside of a venv
- **Pre-flight checks** - Launch script runs comprehensive tests before starting

## Original Fooocus

For all other documentation, features, and usage instructions, please refer to the original [Fooocus repository](https://github.com/lllyasviel/Fooocus).

This fork only modifies the installation and launch process for RTX 5090 compatibility. All image generation features remain the same.

## License

Same as original Fooocus - [GPL-3.0 license](https://github.com/lllyasviel/Fooocus/blob/main/LICENSE)