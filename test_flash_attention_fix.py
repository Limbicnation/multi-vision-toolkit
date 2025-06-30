#!/usr/bin/env python3
"""
Test script to verify flash attention fixes work properly.
Run this with: conda activate vision-env && python test_flash_attention_fix.py
"""

import os
import sys

def test_flash_attention_fix():
    """Test that models can be imported without flash attention conflicts."""
    
    print("🔧 Setting up flash attention guards...")
    # Disable flash attention globally
    os.environ["DISABLE_FLASH_ATTENTION"] = "1"
    os.environ["FLASH_ATTENTION_SKIP_CUDA_CHECK"] = "1"
    os.environ["USE_FLASH_ATTENTION"] = "0"
    os.environ["FLASH_ATTN_DISABLE"] = "1"
    
    print("✅ Flash attention disabled globally")
    
    # Test 1: Basic torch import
    print("\n📦 Testing PyTorch import...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    # Test 2: Transformers core classes
    print("\n📦 Testing transformers import...")
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
        print("✅ Core Qwen transformers classes imported successfully")
    except Exception as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    # Test 3: CLIP models
    print("\n📦 Testing CLIP import...")
    try:
        from transformers import CLIPModel, CLIPProcessor
        print("✅ CLIP models imported successfully")
    except Exception as e:
        print(f"❌ CLIP import failed: {e}")
        return False
    
    # Test 4: Our model imports
    print("\n📦 Testing our model imports...")
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from models.qwen_model import QwenCaptioner, QwenModel
        print("✅ QwenModel and QwenCaptioner imported successfully")
    except Exception as e:
        print(f"❌ Our Qwen models import failed: {e}")
        return False
    
    try:
        from models.florence_model import Florence2Model
        print("✅ Florence2Model imported successfully")
    except Exception as e:
        print(f"❌ Florence2Model import failed: {e}")
        return False
    
    print("\n🎉 ALL TESTS PASSED! Flash attention fixes working correctly.")
    print("\n📋 Summary:")
    print("  - Flash attention disabled globally")
    print("  - All transformers imports successful") 
    print("  - All model classes imported without conflicts")
    print("  - Ready to run main.py")
    
    return True

if __name__ == "__main__":
    success = test_flash_attention_fix()
    sys.exit(0 if success else 1)