#!/bin/bash
set -e

# RTX 5090 optimized launch script
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_MODULE_LOADING=LAZY
export USE_RTX5090=true

# Check UV installation
if ! command -v uv &> /dev/null; then
    echo "ERROR: UV is not installed!"
    echo "Please install UV first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "ERROR: Fooocus environment not found!"
    echo "Please run ./install_fooocus_rtx5090.sh first"
    exit 1
fi

# Double check we're about to activate the right venv
if [ "$VIRTUAL_ENV" != "" ] && [ "$VIRTUAL_ENV" != "$(pwd)/.venv" ]; then
    echo "ERROR: Already in a different virtual environment: $VIRTUAL_ENV"
    echo "Please deactivate it first with 'deactivate'"
    exit 1
fi

# Skip requirements check since we manage dependencies separately
export SKIP_INSTALL=1

# Activate the correct venv
source .venv/bin/activate

# Run pre-flight checks unless SKIP_CHECKS is set
if [ "${SKIP_CHECKS}" != "1" ]; then
    echo "Running RTX 5090 pre-flight checks..."
    if ! python test_rtx5090_setup.py; then
        echo ""
        echo "❌ Pre-flight checks failed!"
        echo "Please fix the issues above before launching Fooocus."
        echo "To skip checks (not recommended): SKIP_CHECKS=1 $0"
        exit 1
    fi
    
    echo ""
    echo "✅ Pre-flight checks passed!"
    echo ""
else
    echo "⚠️  Skipping pre-flight checks (SKIP_CHECKS=1)"
fi

# Launch with python directly (already in correct venv)
echo "Launching Fooocus with RTX 5090 optimizations..."
exec python launch.py \
    --listen 127.0.0.1 \
    --port 7865 \
    --async-cuda-allocation \
    --gpu-device-id 0 \
    "$@"