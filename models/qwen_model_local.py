# models/qwen_model_local.py
from models.qwen_model import QwenModel
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class QwenModelLocal(QwenModel):
    """
    Extension of QwenModel that explicitly loads and saves models to the local ./models/ directory
    instead of relying on Hugging Face's default caching mechanism.
    """
    
    def __init__(self, model_path: str = None, local_models_dir: str = None):
        """
        Initialize a QwenModel that loads from and saves to a local directory.
        
        Args:
            model_path: Override for the default model path
            local_models_dir: Directory to store model files, defaults to ./models/weights/ if not specified
        """
        self.local_models_dir = local_models_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "models", "weights"
        )
        logger.info(f"Initializing QwenModelLocal with local storage directory: {self.local_models_dir}")
        
        # Create the local models directory if it doesn't exist
        os.makedirs(self.local_models_dir, exist_ok=True)
        
        # If model_path is not specified, use a subdirectory for the default Qwen model
        self.model_name = "Qwen2.5-VL-3B-Instruct"
        self.local_model_path = os.path.join(self.local_models_dir, self.model_name)
        
        # Set up environment variables to ensure HF uses our local directory
        self._setup_environment()
        
        # Call parent's __init__ with our local path if the model exists locally
        if os.path.exists(self.local_model_path) and self._check_model_files():
            logger.info(f"Using local model from: {self.local_model_path}")
            super().__init__(model_path=self.local_model_path)
        else:
            logger.info(f"Local model not found at {self.local_model_path}. Will download to this location.")
            # Set up environment to download to our local path
            self._prepare_for_download()
            super().__init__(model_path="Qwen/Qwen2.5-VL-3B-Instruct")
            # After downloading, move files if needed
            self._ensure_model_locally_saved()
    
    def _setup_environment(self):
        """Set environment variables to ensure models are saved locally and properly"""
        # Set cache directory to our local directory
        os.environ["TRANSFORMERS_CACHE"] = self.local_models_dir
        os.environ["HF_HOME"] = self.local_models_dir
        os.environ["HF_HUB_CACHE"] = self.local_models_dir
        
        # Force local offline mode if we already have the model
        if os.path.exists(self.local_model_path) and self._check_model_files():
            logger.info("Setting HF_HUB_OFFLINE=1 to force using local files")
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            # Ensure we're in online mode for downloading
            if "HF_HUB_OFFLINE" in os.environ:
                del os.environ["HF_HUB_OFFLINE"]
    
    def _check_model_files(self) -> bool:
        """
        Check if the local model directory has the necessary files for the model to load.
        
        Returns:
            bool: True if the model appears to be complete, False otherwise
        """
        required_files = ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"]
        model_files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        
        # Check for essential metadata files
        for file in required_files:
            if not os.path.exists(os.path.join(self.local_model_path, file)):
                logger.warning(f"Missing required file: {file}")
                return False
        
        # Check for model weight files (at least some should be present)
        found_model_files = False
        for file in model_files:
            if os.path.exists(os.path.join(self.local_model_path, file)):
                found_model_files = True
                break
        
        if not found_model_files:
            logger.warning("No model weight files found")
            return False
            
        return True
    
    def _prepare_for_download(self):
        """Prepare environment for downloading the model to our local directory"""
        logger.info(f"Preparing to download Qwen model to {self.local_model_path}")
        os.makedirs(self.local_model_path, exist_ok=True)
        
        # Override HF's cache dirs to point to our local path
        from huggingface_hub import constants
        
        # Save original attributes to restore later if needed
        self._original_hf_hub_cache = constants.HF_HUB_CACHE
        self._original_default_cache_path = constants.default_cache_path
        
        # Monkey patch HF constants to ensure downloading to our location
        constants.HF_HUB_CACHE = Path(self.local_models_dir)
        constants.default_cache_path = Path(self.local_models_dir)
    
    def _ensure_model_locally_saved(self):
        """
        Verify that the model was properly saved to the local directory.
        If not, attempt to copy or move it from the HF cache.
        """
        model_dir = os.path.join(self.local_models_dir, self.model_name)
        
        # Check if model exists in our target location
        if not self._check_model_files():
            logger.warning(f"Model not correctly saved to {model_dir}, attempting to locate in HF cache...")
            
            # Look in common HF cache locations
            cache_candidates = [
                os.path.expanduser("~/.cache/huggingface/hub"),
                os.path.join(Path.home(), ".cache", "huggingface"),
                os.environ.get("HF_HOME", ""),
                os.environ.get("TRANSFORMERS_CACHE", "")
            ]
            
            # Try to locate model files in HF cache
            found_model_location = None
            for cache_dir in cache_candidates:
                if not cache_dir:
                    continue
                
                # Search for model directory in the cache
                for root, dirs, _ in os.walk(cache_dir):
                    for d in dirs:
                        if d.replace("-", "/") == "Qwen/Qwen2.5-VL-3B-Instruct":
                            found_model_location = os.path.join(root, d)
                            break
                    if found_model_location:
                        break
                if found_model_location:
                    break
            
            # If we found the model in the cache, copy it
            if found_model_location:
                logger.info(f"Found model in cache at: {found_model_location}, copying to local directory...")
                self._copy_model_files(found_model_location, model_dir)
            else:
                logger.warning("Could not find model files in HF cache. Please try downloading with fix_awq_qwen.py")
    
    def _copy_model_files(self, source_dir: str, target_dir: str):
        """
        Copy model files from source to target directory.
        
        Args:
            source_dir: Source directory containing model files
            target_dir: Target directory to copy files to
        """
        import shutil
        os.makedirs(target_dir, exist_ok=True)
        
        # List all files in source directory
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            target_item = os.path.join(target_dir, item)
            
            if os.path.isdir(source_item):
                # Recursively copy subdirectories
                shutil.copytree(source_item, target_item, dirs_exist_ok=True)
            else:
                # Copy files
                shutil.copy2(source_item, target_item)
                logger.info(f"Copied {item} to {target_dir}")
    
    def _setup_model(self) -> None:
        """
        Override parent's _setup_model to ensure we use the local model path.
        """
        # If the local model exists, use it directly
        if os.path.exists(self.local_model_path) and self._check_model_files():
            # Temporarily override model_path to use local path
            original_model_path = self.model_path
            self.model_path = self.local_model_path
            logger.info(f"Setting up model from local path: {self.local_model_path}")
            
            try:
                # Call parent's _setup_model method
                super()._setup_model()
            finally:
                # Restore original model_path
                self.model_path = original_model_path
        else:
            # If local model doesn't exist, let parent handle it as usual
            # but ensure we're saving to the local directory
            super()._setup_model()
            
            # After setup, ensure the model is saved locally
            self._ensure_model_locally_saved()