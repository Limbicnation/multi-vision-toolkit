#!/usr/bin/env python3
"""
Environment check script for Multi-Vision Toolkit
This script helps diagnose dependency issues and environment setup problems.
"""

import sys
import os

def check_python_version():
    """Check Python version compatibility."""
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ is required")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def check_conda_env():
    """Check if we're in the right conda environment."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    print(f"Current conda environment: {conda_env}")
    if conda_env == 'vision-env':
        print("✅ Running in vision-env environment")
        return True
    elif conda_env == 'base':
        print("❌ Running in base environment. Please activate vision-env:")
        print("   conda activate vision-env")
        return False
    else:
        print(f"❌ Running in {conda_env} environment. Please activate vision-env:")
        print("   conda activate vision-env")
        return False

def check_pytorch():
    """Check PyTorch installation."""
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
        return True
    except ImportError:
        print("❌ PyTorch not found. Please install with:")
        print("   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126")
        return False

def check_transformers():
    """Check transformers installation."""
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
        
        # Check for specific classes
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
            print("✅ AutoModelForImageTextToText available")
            return True
        except ImportError as e:
            print(f"❌ AutoModelForImageTextToText not available: {e}")
            print("   Please update transformers: pip install transformers>=4.46.0")
            return False
    except ImportError:
        print("❌ Transformers not found. Please install with:")
        print("   pip install transformers>=4.46.0")
        return False

def check_qwen_utils():
    """Check qwen_vl_utils installation."""
    try:
        from qwen_vl_utils import process_vision_info
        print("✅ qwen_vl_utils available")
        return True
    except ImportError:
        print("❌ qwen_vl_utils not found. Please install with:")
        print("   pip install qwen-vl-utils[decord]==0.0.8")
        return False

def check_other_deps():
    """Check other important dependencies."""
    deps = [
        ('PIL', 'Pillow'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('cv2', 'opencv-python'),
        ('accelerate', 'accelerate'),
        ('bitsandbytes', 'bitsandbytes'),
    ]
    
    all_good = True
    for module, package in deps:
        try:
            __import__(module)
            print(f"✅ {module} available")
        except ImportError:
            print(f"❌ {module} not found. Install with: pip install {package}")
            all_good = False
    
    return all_good

def main():
    """Main check function."""
    print("=" * 50)
    print("Multi-Vision Toolkit Environment Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Conda Environment", check_conda_env),
        ("PyTorch", check_pytorch),
        ("Transformers", check_transformers),
        ("Qwen Utils", check_qwen_utils),
        ("Other Dependencies", check_other_deps),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 20)
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! Your environment is ready.")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nQuick fix steps:")
        print("1. conda activate vision-env")
        print("2. pip install -r requirements.txt")
        print("3. pip install qwen-vl-utils[decord]==0.0.8")
        print("4. ./clone_models.sh  # if you haven't downloaded models yet")
    
    return all_passed

if __name__ == "__main__":
    main()