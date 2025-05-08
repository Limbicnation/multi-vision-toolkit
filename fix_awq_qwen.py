#!/usr/bin/env python
"""
Fix script for AWQ Qwen model in Florence2-Vision-Toolkit.
This script will try to fix AWQ module loading by switching to non-AWQ model version.
"""

import os
import sys
import logging
import torch
import platform
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_awq")

def setup_cache_directory():
    """Setup persistent cache directory with cross-platform support."""
    home_dir = Path.home()
    
    # Determine platform-specific cache location
    if sys.platform == "win32":
        # Windows standard
        cache_dir = home_dir / "AppData" / "Local" / "florence2-vision-toolkit"
    elif sys.platform == "darwin":
        # macOS standard
        cache_dir = home_dir / "Library" / "Caches" / "florence2-vision-toolkit"
    else:
        # Linux and WSL use the same pattern
        cache_dir = home_dir / ".cache" / "florence2-vision-toolkit"
    
    # Create cache directories
    cache_dir.mkdir(parents=True, exist_ok=True)
    transformers_cache = cache_dir / "transformers"
    torch_cache = cache_dir / "torch"
    hf_cache = cache_dir / "huggingface"
    
    # Ensure all subdirectories exist
    transformers_cache.mkdir(exist_ok=True)
    torch_cache.mkdir(exist_ok=True)
    hf_cache.mkdir(exist_ok=True)
    
    # Set environment variables
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["TORCH_HOME"] = str(torch_cache)
    os.environ["HF_HOME"] = str(hf_cache)
    
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
    """Fix Qwen model by downloading non-AWQ version with improved platform compatibility."""
    try:
        from transformers import AutoModel, AutoProcessor
        
        # Setup environment with better error handling
        token = load_token_from_env()
        if not token:
            logger.warning("No HuggingFace token found. Some models may require authentication.")
            logger.info("Consider creating a token at https://huggingface.co/settings/tokens")
        
        cache_dir = setup_cache_directory()
        
        # Check system capabilities
        logger.info(f"Running on platform: {sys.platform}")
        if sys.platform == "linux" and "microsoft" in os.uname().release:
            logger.info("Detected Windows Subsystem for Linux (WSL)")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logger.info(f"CUDA is available: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("CUDA is not available. Using CPU mode which will be much slower.")
        
        # Check AWQ support with more detailed errors
        awq_supported = check_awq_support()
        logger.info(f"AWQ support: {awq_supported}")
        
        if not awq_supported:
            logger.info("AWQ not supported properly, will try to download non-AWQ model")
            logger.info("To enable AWQ support, run: pip install 'autoawq>=0.1.8'")
        
        # First try to download and register the model configuration only
        # This avoids large downloads if just checking availability
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        logger.info(f"Attempting to download {model_name} configuration")
        
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # First check if the model is available without downloading
            try:
                model_info = api.model_info(model_name)
                logger.info(f"Model size: {model_info.safetensors_size / 1e9:.2f} GB")
            except Exception as info_err:
                logger.warning(f"Could not fetch model info: {str(info_err)}")
            
            # Try to download the processor (smaller) first
            logger.info(f"Downloading {model_name} processor")
            processor = AutoProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True,
                cache_dir=os.environ.get("TRANSFORMERS_CACHE")
            )
            logger.info(f"Successfully downloaded {model_name} processor")
            
            # Just initialize model metadata without full loading to save memory
            logger.info(f"Initializing {model_name} model metadata (not loading weights)")
            model_meta = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if cuda_available else torch.float32,
                device_map=None,  # Don't load the model into memory yet
                low_cpu_mem_usage=True,
                cache_dir=os.environ.get("TRANSFORMERS_CACHE")
            )
            logger.info(f"Successfully initialized {model_name} model metadata")
            logger.info(f"Model files should now be downloaded to {cache_dir}/transformers")
            
            # Clean up to free memory
            del model_meta
            del processor
            if cuda_available:
                torch.cuda.empty_cache()
            
            # Alternative model path for git clone method
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "weights")
            if os.path.exists(model_path):
                logger.info(f"Local model directory exists at {model_path}")
                if sys.platform == "win32":
                    clone_cmd = f"cd {model_path} && git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct"
                else:
                    clone_cmd = f"cd {model_path} && git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct"
                logger.info(f"You can clone the model directly using: {clone_cmd}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            
            # Provide platform-specific troubleshooting
            if "CUDA" in str(e) or "cuda" in str(e).lower():
                logger.error("CUDA error detected. Suggestions:")
                logger.error("1. Update your GPU drivers")
                logger.error("2. Try CPU mode if GPU doesn't have enough memory")
            elif "trust_remote_code" in str(e):
                logger.error("Remote code trust issue. Add trust_remote_code=True")
            elif "disk space" in str(e).lower() or "no space" in str(e).lower():
                logger.error("Possible disk space issue. Check free space in your cache directory")
            
            return False
    
    except Exception as e:
        logger.error(f"Error in fix_qwen: {str(e)}")
        # Show Python and crucial package versions for debugging
        logger.info(f"Python version: {sys.version}")
        try:
            import transformers
            logger.info(f"Transformers version: {transformers.__version__}")
        except:
            logger.error("Transformers package not found or has import errors")
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