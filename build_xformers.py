#!/usr/bin/env python3
"""
Build xformers from source for RTX 5090 (SM 120) support.
This is required because pre-built wheels don't support the new architecture.
"""

import os
import sys
import subprocess
import shutil
import tempfile

def run_command(cmd, cwd=None):
    """Run a command and handle errors."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result.stdout

def build_xformers():
    """Build xformers from source."""
    print("Building xformers for RTX 5090 (SM 120) support...")
    
    # Set environment variables
    env = os.environ.copy()
    env['TORCH_CUDA_ARCH_LIST'] = '12.0+PTX'
    env['FORCE_CUDA'] = '1'
    env['MAX_JOBS'] = '4'
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Working in temporary directory: {tmpdir}")
        
        # Clone xformers
        print("Cloning xformers repository...")
        run_command(
            "git clone --depth 1 --branch v0.0.30 https://github.com/facebookresearch/xformers.git",
            cwd=tmpdir
        )
        
        xformers_dir = os.path.join(tmpdir, "xformers")
        
        # Install build dependencies
        print("Installing build dependencies...")
        run_command(f"{sys.executable} -m pip install ninja", cwd=xformers_dir)
        
        # Build and install
        print("Building xformers (this may take 15-30 minutes)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-v", "."],
            cwd=xformers_dir,
            env=env,
            check=True
        )
    
    print("xformers build complete!")

def verify_xformers():
    """Verify xformers installation."""
    try:
        import xformers
        import torch
        
        print(f"\nxformers version: {xformers.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        
        # Test memory efficient attention
        if torch.cuda.is_available():
            print("\nTesting memory efficient attention...")
            from xformers.ops import memory_efficient_attention
            
            # Create test tensors
            batch_size, seq_len, dim = 2, 64, 512
            device = torch.device("cuda")
            dtype = torch.float16
            
            q = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
            k = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
            v = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
            
            # Run attention
            out = memory_efficient_attention(q, k, v)
            print(f"Memory efficient attention output shape: {out.shape}")
            print("xformers is working correctly!")
        else:
            print("Warning: CUDA not available, skipping GPU tests")
            
    except ImportError as e:
        print(f"Error: Failed to import xformers: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during verification: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 12):
        print("Error: Python 3.12+ is required")
        sys.exit(1)
    
    # Check if torch is installed
    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. xformers will be built for CPU only.")
    except ImportError:
        print("Error: PyTorch must be installed before building xformers")
        print("Run: pip install torch==2.9.0.dev20250726+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128")
        sys.exit(1)
    
    build_xformers()
    verify_xformers()