#!/usr/bin/env python3
"""
Test script to verify RTX 5090 GPU functionality with Fooocus
"""

import torch
import sys
import time

def test_gpu():
    """Test GPU functionality and performance."""
    print("="*60)
    print("RTX 5090 GPU Test for Fooocus")
    print("="*60)
    
    # Check PyTorch version
    print(f"\nPyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ CUDA is not available!")
        print("Please ensure you have installed the CUDA 12.8 compatible PyTorch.")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Get GPU information
    device_count = torch.cuda.device_count()
    print(f"\nNumber of GPUs: {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Multiprocessors: {props.multi_processor_count}")
        
        # Check for RTX 5090
        if "5090" in torch.cuda.get_device_name(i):
            print("  ✅ RTX 5090 detected!")
            if props.major == 12 and props.minor == 0:
                print("  ✅ SM 120 compute capability confirmed!")
            else:
                print(f"  ⚠️  Unexpected compute capability: {props.major}.{props.minor}")
    
    # Test memory allocation
    print("\n" + "="*40)
    print("Testing memory allocation...")
    try:
        device = torch.device("cuda:0")
        
        # Allocate 1GB tensor
        size = 1024 * 1024 * 1024 // 4  # 1GB of float32
        tensor = torch.zeros(size, dtype=torch.float32, device=device)
        print("✅ Successfully allocated 1GB tensor")
        
        # Test computation
        print("\nTesting computation performance...")
        
        # FP32 benchmark
        matrix_size = 4096
        a = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
        b = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
        
        # Warmup
        for _ in range(5):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        iterations = 50
        for _ in range(iterations):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        tflops_fp32 = (2 * matrix_size**3 * iterations) / (elapsed * 1e12)
        print(f"✅ FP32 performance: {tflops_fp32:.1f} TFLOPS")
        
        # FP16 benchmark
        a_fp16 = a.half()
        b_fp16 = b.half()
        
        # Warmup
        for _ in range(5):
            c_fp16 = torch.matmul(a_fp16, b_fp16)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(iterations):
            c_fp16 = torch.matmul(a_fp16, b_fp16)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        tflops_fp16 = (2 * matrix_size**3 * iterations) / (elapsed * 1e12)
        print(f"✅ FP16 performance: {tflops_fp16:.1f} TFLOPS")
        
        # RTX 5090 should achieve ~102 TFLOPS in FP16
        if tflops_fp16 > 80:
            print("✅ Performance is consistent with RTX 5090!")
        
        # Cleanup
        del tensor, a, b, c, a_fp16, b_fp16, c_fp16
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    
    # Test xformers if available
    print("\n" + "="*40)
    print("Testing xformers...")
    try:
        import xformers
        from xformers.ops import memory_efficient_attention
        
        print(f"✅ xformers version: {xformers.__version__}")
        
        # Test memory efficient attention
        batch_size, seq_len, dim = 2, 512, 768
        q = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float16)
        k = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float16)
        
        out = memory_efficient_attention(q, k, v)
        print(f"✅ Memory efficient attention working (output shape: {out.shape})")
        
    except ImportError:
        print("⚠️  xformers not installed. Run build_xformers.py to build from source.")
    except Exception as e:
        print(f"❌ xformers error: {e}")
    
    # Test environment variables
    print("\n" + "="*40)
    print("Checking environment variables...")
    
    env_vars = {
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'TORCH_CUDNN_V8_API_ENABLED': '1',
        'CUDA_MODULE_LOADING': 'LAZY'
    }
    
    import os
    for var, expected in env_vars.items():
        actual = os.environ.get(var, 'Not set')
        if actual == expected:
            print(f"✅ {var} = {actual}")
        else:
            print(f"⚠️  {var} = {actual} (expected: {expected})")
    
    print("\n" + "="*60)
    print("GPU test completed successfully!")
    print("Your RTX 5090 is ready for Fooocus!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = test_gpu()
    sys.exit(0 if success else 1)