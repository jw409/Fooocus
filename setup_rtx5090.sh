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

# Create virtual environment
echo ""
echo "Creating Python 3.12 virtual environment..."
uv venv --python 3.12 venv_rtx5090

# Activate virtual environment
source venv_rtx5090/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.8
echo ""
echo "Installing PyTorch 2.9.0 with CUDA 12.8 support..."
pip install torch==2.9.0.dev20250726+cu128 torchvision==0.21.0.dev20250726+cu128 torchaudio==2.9.0.dev20250726+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128

# Install CUDA runtime libraries
echo ""
echo "Installing CUDA 12.8 runtime libraries..."
pip install nvidia-cuda-runtime-cu12>=12.8.0 nvidia-cudnn-cu12>=9.5.0 nvidia-cublas-cu12>=12.8.0 nvidia-curand-cu12>=10.3.7 nvidia-cusolver-cu12>=11.7.1 nvidia-cusparse-cu12>=12.6.0 nvidia-cufft-cu12>=11.3.0

# Build xformers from source
echo ""
echo "Building xformers from source for SM 120 support..."
echo "This may take 15-30 minutes..."

# Set environment variables for xformers build
export TORCH_CUDA_ARCH_LIST='12.0+PTX'
export FORCE_CUDA=1
export MAX_JOBS=4

# Clone and build xformers
if [ -d "xformers_build" ]; then
    rm -rf xformers_build
fi
git clone https://github.com/facebookresearch/xformers.git xformers_build
cd xformers_build
git checkout v0.0.30
pip install -r requirements.txt
pip install -e .
cd ..

# Install remaining dependencies
echo ""
echo "Installing remaining dependencies..."
pip install -r requirements_rtx5090.txt

# Create optimized launch script
echo ""
echo "Creating optimized launch script..."
cat > launch_rtx5090.sh << 'EOF'
#!/bin/bash

# RTX 5090 optimized launch script for Fooocus

# Activate virtual environment
source venv_rtx5090/bin/activate

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_MODULE_LOADING=LAZY

# Launch Fooocus with RTX 5090 optimizations
python launch.py \
    --xformers \
    --opt-sdp-attention \
    --no-half-vae \
    --cuda-device 0 \
    --cuda-stream \
    "$@"
EOF

chmod +x launch_rtx5090.sh

# Test GPU detection
echo ""
echo "Testing GPU detection..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo ""
echo "==================================================="
echo "Setup complete!"
echo ""
echo "To run Fooocus with RTX 5090 optimizations:"
echo "  ./launch_rtx5090.sh"
echo ""
echo "Or activate the environment manually:"
echo "  source venv_rtx5090/bin/activate"
echo "  python launch.py"
echo "==================================================="