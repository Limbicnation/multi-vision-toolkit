#!/usr/bin/env python
"""
Fix script for AWQ Qwen model in Florence2-Vision-Toolkit.
This script will try to fix AWQ module loading by switching to non-AWQ model version.
"""

import os
import sys
import logging
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_awq")

def setup_cache_directory():
    """Setup persistent cache directory."""
    home_dir = Path.home()
    cache_dir = home_dir / ".cache" / "florence2-vision-toolkit"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    logger.info(f"Set up persistent cache at: {cache_dir}")
    return cache_dir

def load_token_from_env():
    """Load HuggingFace token from .env file or environment variable."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('HF_TOKEN='):
                        token = line.split('=', 1)[1]
                        token = token.strip('\'"')
                        break
    
    if token:
        logger.info("HuggingFace token found")
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        os.environ["HF_HUB_TOKEN"] = token
        return token
    else:
        logger.warning("No HuggingFace token found")
        return None

def check_awq_support():
    """Check if AWQ is properly supported."""
    try:
        import autoawq
        logger.info(f"autoawq version: {autoawq.__version__}")
        
        try:
            from transformers import AwqConfig
            logger.info("transformers has AwqConfig support")
            return True
        except ImportError:
            logger.warning("transformers doesn't support AwqConfig")
            return False
    except ImportError:
        logger.warning("autoawq package not found")
        return False

def fix_qwen():
    """Fix Qwen model by downloading non-AWQ version."""
    try:
        from transformers import AutoModel, AutoProcessor
        
        # Setup environment
        load_token_from_env()
        setup_cache_directory()
        
        # Check AWQ support
        awq_supported = check_awq_support()
        logger.info(f"AWQ support: {awq_supported}")
        
        if not awq_supported:
            logger.info("AWQ not supported properly, will try to download non-AWQ model")
        
        # Try to download non-AWQ model
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        logger.info(f"Attempting to download {model_name}")
        
        try:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            logger.info(f"Successfully downloaded {model_name} processor")
            
            # Just initialize model metadata without full loading to save memory
            model_meta = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=None,  # Don't load the model into memory yet
                low_cpu_mem_usage=True
            )
            logger.info(f"Successfully initialized {model_name} model metadata")
            logger.info("Model files should now be downloaded for future use")
            
            # Clean up to free memory
            del model_meta
            del processor
            torch.cuda.empty_cache()
            
            # Alternative model path
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "weights")
            if os.path.exists(model_path):
                logger.info(f"Local model directory exists at {model_path}")
                logger.info("You can clone the model directly using:")
                logger.info(f"  cd {model_path} && git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            return False
    
    except Exception as e:
        logger.error(f"Error in fix_qwen: {str(e)}")
        return False

if __name__ == "__main__":
    print("Florence2 Vision Toolkit - Fix AWQ Qwen model")
    print("---------------------------------------------")
    
    if fix_qwen():
        print("\nSuccess! The non-AWQ model has been downloaded.")
        print("You can now run the application with:")
        print("python main.py --review_dir data/review --approved_dir data/approved --rejected_dir data/rejected --model qwen")
        sys.exit(0)
    else:
        print("\nFailed to fix Qwen model.")
        print("Try installing the CLIP model as a fallback:")
        print("pip install git+https://github.com/openai/CLIP.git")
        sys.exit(1)