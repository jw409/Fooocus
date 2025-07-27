#!/usr/bin/env python3
"""
RTX 5090 optimized launcher for Fooocus
Automatically sets environment variables and launches with optimal settings
"""

import os
import sys

# Set RTX 5090 environment variables
os.environ['USE_RTX5090'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

# Add RTX 5090 specific arguments
rtx5090_args = [
    '--xformers',
    '--opt-sdp-attention',
    '--no-half-vae',
    '--cuda-stream'
]

# Combine with user arguments
import launch
sys.argv.extend(rtx5090_args)

# Run the main launch script
if __name__ == '__main__':
    launch