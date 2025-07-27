#!/bin/bash
set -e

echo "==================================================="
echo "Fooocus RTX 5090 Setup Script"
echo "==================================================="
echo "This script will set up Fooocus with CUDA 12.8 support"
echo "for RTX 5090 and other RTX 50-series GPUs"
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Error: This script is designed for Linux systems."
    echo "For Windows, please use setup_rtx5090.bat"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]); then
    echo "Error: Python 3.12 or higher is required."
    echo "Current version: $PYTHON_VERSION"
    echo "Please install Python 3.12+ and try again."
    exit 1
fi

# Install UV if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "UV is already installed"
fi

# Check for pyproject.toml
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found."
    echo "Please run this script from the Fooocus root directory."
    exit 1
fi

# Create UV-managed environment and sync dependencies
echo ""
echo "Setting up UV project environment..."
echo "UV will create and manage the virtual environment automatically"

# Set up UV to use the PyTorch nightly index for CUDA 12.8
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/nightly/cu128"

# Sync the project - this creates .venv and installs all dependencies
echo ""
echo "Installing all dependencies via UV..."
uv sync

# Build xformers from source in the UV environment
echo ""
echo "Building xformers from source for SM 120 support..."
echo "This may take 15-30 minutes..."

# Set environment variables for xformers build
export TORCH_CUDA_ARCH_LIST='12.0+PTX'
export FORCE_CUDA=1
export MAX_JOBS=4

# Use UV to run the xformers build script
uv run python build_xformers.py

# Create optimized launch script that uses UV
echo ""
echo "Creating optimized launch script..."
cat > launch_rtx5090.sh << 'EOF'
#!/bin/bash

# RTX 5090 optimized launch script for Fooocus using UV

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_MODULE_LOADING=LAZY
export USE_RTX5090=true

# Launch Fooocus with RTX 5090 optimizations using UV
uv run python launch.py \
    --xformers \
    --opt-sdp-attention \
    --no-half-vae \
    --cuda-device 0 \
    --cuda-stream \
    "$@"
EOF

chmod +x launch_rtx5090.sh

# Test GPU detection using UV
echo ""
echo "Testing GPU detection..."
uv run python test_gpu.py

echo ""
echo "==================================================="
echo "Setup complete!"
echo ""
echo "To run Fooocus with RTX 5090 optimizations:"
echo "  ./launch_rtx5090.sh"
echo ""
echo "Or use UV directly:"
echo "  uv run python launch.py"
echo ""
echo "UV has created a .venv directory with all dependencies."
echo "==================================================="