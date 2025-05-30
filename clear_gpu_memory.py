#!/usr/bin/env python3
"""
GPU Memory Cleanup Script for QwenCaptioner OOM issues
"""

import os
import gc
import logging
import subprocess
import sys

# Set PyTorch memory allocation config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    import torch
    if torch.cuda.is_available():
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        logger.info("🧹 Starting aggressive GPU memory cleanup...")
        
        # Show initial memory status
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        
        total_gb = total_memory / (1024**3)
        allocated_gb_before = allocated_before / (1024**3)
        reserved_gb_before = reserved_before / (1024**3)
        free_gb_before = (total_memory - reserved_before) / (1024**3)
        
        logger.info(f"📊 Before cleanup - Total: {total_gb:.2f}GB, Allocated: {allocated_gb_before:.2f}GB, Reserved: {reserved_gb_before:.2f}GB, Free: {free_gb_before:.2f}GB")
        
        # Aggressive cleanup
        for i in range(5):
            logger.info(f"🔄 Cleanup round {i+1}/5...")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Final memory status
        allocated_after = torch.cuda.memory_allocated()
        reserved_after = torch.cuda.memory_reserved()
        
        allocated_gb_after = allocated_after / (1024**3)
        reserved_gb_after = reserved_after / (1024**3)
        free_gb_after = (total_memory - reserved_after) / (1024**3)
        
        freed_gb = reserved_gb_before - reserved_gb_after
        
        logger.info(f"📊 After cleanup - Allocated: {allocated_gb_after:.2f}GB, Reserved: {reserved_gb_after:.2f}GB, Free: {free_gb_after:.2f}GB")
        logger.info(f"✅ Freed {freed_gb:.2f}GB of GPU memory!")
        
        if free_gb_after < 4.0:
            logger.warning("⚠️  Still insufficient memory for 7B model. Consider:")
            logger.warning("   1. Kill other GPU processes: nvidia-smi then kill -9 <pid>")
            logger.warning("   2. Restart your environment")
            logger.warning("   3. Use a smaller model")
            
            # Show GPU processes
            try:
                result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    logger.info("🔍 Current GPU processes:")
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            logger.info(f"   {line}")
            except:
                pass
        else:
            logger.info(f"🎉 Memory cleanup successful! {free_gb_after:.2f}GB should be enough for 4-bit quantized 7B model.")
            
    else:
        print("❌ CUDA not available")
        
except ImportError:
    print("❌ PyTorch not installed")
    sys.exit(1)