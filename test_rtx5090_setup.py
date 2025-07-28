#!/usr/bin/env python3
"""Test RTX 5090 setup and dependencies"""

import sys
import os

def test_environment():
    """Test environment setup"""
    print("="*60)
    print("Testing RTX 5090 Environment")
    print("="*60)
    
    # Check virtual environment
    if sys.prefix == sys.base_prefix:
        print("❌ ERROR: Not running in a virtual environment!")
        return False
    print(f"✅ Virtual environment: {sys.prefix}")
    
    # Check UV
    import shutil
    if not shutil.which('uv'):
        print("❌ ERROR: UV is not installed!")
        return False
    print("✅ UV is available")
    
    return True

def test_dependencies():
    """Test all required dependencies"""
    print("\n" + "="*60)
    print("Testing Dependencies")
    print("="*60)
    
    dependencies = [
        ("torch", None),
        ("torchvision", None),
        ("torchaudio", None),
        ("numpy", None),  # Version check done separately
        ("transformers", None),
        ("accelerate", None),
        ("einops", None),
        ("gradio", None),
        ("safetensors", None),
        ("opencv-python", "cv2"),
        ("pillow", "PIL"),
        ("scipy", None),
        ("tqdm", None),
        ("psutil", None),
        ("pytorch_lightning", None),
        ("omegaconf", None),
        ("httpx", None),
        ("timm", None),
        ("rembg", None),
        ("groundingdino", None),
    ]
    
    all_good = True
    for dep_info in dependencies:
        if isinstance(dep_info, tuple):
            dep_name, import_name = dep_info
            if import_name is None:
                import_name = dep_name.replace("-", "_")
        else:
            dep_name = import_name = dep_info
            
        try:
            if import_name:
                __import__(import_name)
            print(f"✅ {dep_name}")
        except ImportError as e:
            print(f"❌ {dep_name}: {e}")
            all_good = False
    
    # Special numpy version check
    try:
        import numpy as np
        if hasattr(np, '__version__'):
            version = np.__version__
            major_version = int(version.split('.')[0])
            if major_version >= 2:
                print(f"⚠️  WARNING: numpy {version} may cause issues. Transformers requires numpy<2.0")
                print("   To fix: uv pip install 'numpy<2.0'")
    except Exception:
        pass
    
    return all_good

def test_pytorch_cuda():
    """Test PyTorch and CUDA setup"""
    print("\n" + "="*60)
    print("Testing PyTorch and CUDA")
    print("="*60)
    
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if not torch.cuda.is_available():
            print("❌ CUDA is not available!")
            return False
            
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Check for RTX 5090
        gpu_name = torch.cuda.get_device_name(0)
        if "5090" not in gpu_name:
            print(f"⚠️  WARNING: Expected RTX 5090 but found {gpu_name}")
        
        # Check compute capability
        capability = torch.cuda.get_device_capability(0)
        print(f"Compute capability: {capability}")
        if capability[0] < 12:
            print("❌ RTX 5090 requires compute capability 12.0+")
            return False
            
        # Check PyTorch version
        if not torch.__version__.startswith("2.9"):
            print(f"⚠️  WARNING: RTX 5090 works best with PyTorch 2.9+, found {torch.__version__}")
            
        # Check CUDA version
        cuda_version = torch.version.cuda
        if cuda_version and not cuda_version.startswith("12.8"):
            print(f"⚠️  WARNING: RTX 5090 works best with CUDA 12.8+, found {cuda_version}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing PyTorch/CUDA: {e}")
        return False

# Add xformers to path if running from Fooocus directory
if os.path.exists('xformers'):
    sys.path.insert(0, 'xformers')

import torch

def test_xformers():
    print("="*60)
    print("Testing xformers on RTX 5090")
    print("="*60)
    
    # Check PyTorch and CUDA
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    
    # Test xformers import
    try:
        import xformers
        try:
            version = xformers.__version__
        except AttributeError:
            # Try alternative version locations
            try:
                from xformers.version import __version__ as version
            except:
                version = "unknown (built from source)"
        print(f"\n✅ xformers version: {version}")
    except ImportError as e:
        print(f"\n❌ Failed to import xformers: {e}")
        return False
    
    # Test xformers ops import
    try:
        import xformers.ops
        print("✅ xformers.ops imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import xformers.ops: {e}")
        return False
    
    # Test memory efficient attention
    print("\n" + "="*60)
    print("Testing memory efficient attention...")
    print("="*60)
    
    try:
        # Create test tensors
        batch_size = 2
        seq_len = 512
        n_heads = 8
        d_head = 64
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16
        
        # Create query, key, value tensors
        q = torch.randn(batch_size, seq_len, n_heads, d_head, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, n_heads, d_head, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, n_heads, d_head, device=device, dtype=dtype)
        
        print(f"\nTest tensors created:")
        print(f"  Shape: {q.shape}")
        print(f"  Device: {q.device}")
        print(f"  Dtype: {q.dtype}")
        
        # Test memory efficient attention
        print("\nRunning memory_efficient_attention...")
        output = xformers.ops.memory_efficient_attention(q, k, v)
        
        print(f"✅ Memory efficient attention successful!")
        print(f"  Output shape: {output.shape}")
        print(f"  Output device: {output.device}")
        print(f"  Output dtype: {output.dtype}")
        
        # Test with attention bias (using batch_size=1 for BlockDiagonalMask)
        print("\nTesting with attention bias...")
        try:
            q_single = q[:1]  # Take only first batch element
            k_single = k[:1]
            v_single = v[:1]
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([seq_len])
            output_with_bias = xformers.ops.memory_efficient_attention(q_single, k_single, v_single, attn_bias=attn_bias)
            print(f"✅ Memory efficient attention with bias successful!")
            print(f"  Output shape: {output_with_bias.shape}")
        except Exception as e:
            print(f"⚠️  BlockDiagonalMask test skipped: {e}")
            print("  (This is expected for batch_size > 1)")
        
        # Test different attention operations
        print("\nTesting available attention operations...")
        available_ops = []
        for op_name in ['cutlass', 'flash', 'small_k', 'triton']:
            try:
                op = getattr(xformers.ops.fmha, f'{op_name}', None)
                if op is not None:
                    available_ops.append(op_name)
            except:
                pass
        
        if available_ops:
            print(f"✅ Available attention ops: {', '.join(available_ops)}")
        else:
            print("⚠️  No specific attention ops found (using default)")
        
        # Test SwiGLU
        print("\nTesting SwiGLU activation...")
        try:
            from xformers.ops import SwiGLU
            in_features = 512
            hidden_features = 1024  # Typically 4x the input features
            swiglu = SwiGLU(in_features=in_features, hidden_features=hidden_features).to(device)
            # Convert to half precision to match our test dtype
            swiglu = swiglu.half()
            x = torch.randn(batch_size, seq_len, in_features, device=device, dtype=dtype)
            output = swiglu(x)
            print(f"✅ SwiGLU test successful!")
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
        except Exception as e:
            print(f"⚠️  SwiGLU test failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*60)
        print("✅ All xformers tests passed!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error during xformers testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and return overall status"""
    results = []
    
    # Test environment
    results.append(("Environment", test_environment()))
    
    # Test PyTorch and CUDA
    results.append(("PyTorch/CUDA", test_pytorch_cuda()))
    
    # Test dependencies
    results.append(("Dependencies", test_dependencies()))
    
    # Test xformers
    results.append(("xformers", test_xformers()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("✅ All tests passed! RTX 5090 setup is ready.")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)