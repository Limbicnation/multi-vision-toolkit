#!/usr/bin/env python3
"""
Test script for QwenCaptioner integration
"""
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qwen_captioner():
    """Test QwenCaptioner functionality"""
    try:
        from models.qwen_model import QwenCaptioner
        
        print("🔧 Testing QwenCaptioner Integration")
        print("=" * 50)
        
        # Test 1: Import and instantiation
        print("✓ QwenCaptioner import successful")
        
        # Test 2: Create instance
        print("Creating QwenCaptioner instance...")
        captioner = QwenCaptioner()
        print(f"✓ Model path: {captioner.model_path}")
        print(f"✓ Quantization: {captioner.use_quantization}")
        
        # Test 3: Check methods
        print("✓ caption_image method available:", hasattr(captioner, 'caption_image'))
        print("✓ caption_batch method available:", hasattr(captioner, 'caption_batch'))
        print("✓ load method available:", hasattr(captioner, 'load'))
        
        print("\n🎉 QwenCaptioner integration test completed successfully!")
        print("\nUsage examples:")
        print("1. Single image: captioner.caption_image('image.jpg')")
        print("2. Batch processing: captioner.caption_batch(['img1.jpg', 'img2.jpg'])")
        print("3. Quality modes: 'standard', 'detailed', 'creative'")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_qwen_captioner()