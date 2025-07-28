#!/bin/bash
set -e

echo "==================================================="
echo "Fooocus RTX 5090 Installation Script"
echo "==================================================="
echo "This script installs Fooocus with pre-built xformers"
echo ""

# Check UV installation
if ! command -v uv &> /dev/null; then
    echo "ERROR: UV is not installed!"
    echo "Please install UV first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if in correct directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found."
    echo "Please run this script from the Fooocus root directory."
    exit 1
fi

# Create/use Fooocus venv (NOT the build venv)
if [ ! -d ".venv" ]; then
    echo "Creating Fooocus virtual environment..."
    uv venv --python 3.12
else
    echo "Using existing .venv"
fi

# Install PyTorch with CUDA 12.8
echo ""
echo "Installing PyTorch with CUDA 12.8 support..."
source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/nightly/cu128 \
    torch==2.9.0.dev20250726+cu128 \
    torchvision==0.24.0.dev20250727+cu128 \
    torchaudio==2.8.0.dev20250727+cu128

# Install base dependencies
echo ""
echo "Installing Fooocus dependencies (without PyTorch)..."
uv pip install \
    torchsde==0.2.6 \
    einops==0.8.0 \
    transformers==4.42.4 \
    safetensors==0.4.3 \
    accelerate==0.32.1 \
    pyyaml==6.0.1 \
    pillow==10.4.0 \
    scipy==1.14.0 \
    tqdm==4.66.4 \
    psutil==6.0.0 \
    pytorch_lightning==2.3.3 \
    omegaconf==2.3.0 \
    gradio==3.41.2 \
    pygit2==1.15.1 \
    opencv-contrib-python-headless==4.10.0.84 \
    httpx==0.27.0 \
    onnxruntime==1.18.1 \
    timm==1.0.7 \
    numpy==1.26.4 \
    tokenizers==0.19.1 \
    packaging==24.1 \
    rembg==2.0.57 \
    groundingdino-py==0.4.0 \
    segment_anything==1.0

# Install xformers if available
echo ""
if [ -f "dist/xformers-0.0.32+8ed0992c.d20250727-cp39-abi3-linux_x86_64.whl" ]; then
    echo "Installing xformers wheel..."
    uv pip install dist/xformers-0.0.32+8ed0992c.d20250727-cp39-abi3-linux_x86_64.whl
elif [ -d "xformers" ]; then
    echo "Installing xformers from source directory..."
    uv pip install ./xformers
else
    echo "WARNING: xformers not found. Run build_xformers_uv.sh first for best performance."
fi

# Verify installation
echo ""
echo "Verifying installation..."
python verify_rtx5090.py || {
    echo "WARNING: Verification failed. Please check the installation."
    exit 1
}

deactivate

echo ""
echo "==================================================="
echo "Installation complete and verified!"
echo ""
echo "To run Fooocus:"
echo "  ./launch_fooocus_rtx5090.sh"
echo "==================================================="