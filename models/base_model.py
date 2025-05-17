# models/base_model.py
from abc import ABC, abstractmethod
import torch
import logging
from typing import Tuple, Optional
from PIL import Image

logger = logging.getLogger(__name__)

class BaseVisionModel(ABC):
    def __init__(self, device=None):
        logger.info("Initializing vision model...")
        
        # Get the device from argument or auto-detect
        self.device = self._get_optimal_device(device)
        
        # Determine optimal tensor dtype based on device capabilities
        self.torch_dtype = self._get_optimal_dtype()
        
        # Print device info for debugging
        self._log_device_info()
            
        logger.info(f"Using device: {self.device} with dtype: {self.torch_dtype}")
        
        # Setup model
        self._setup_model()
        
    def _get_optimal_device(self, requested_device=None):
        """
        Determine the optimal device based on system capabilities and user preference.
        Handles MPS (Apple Silicon), CUDA, CPU, and specific device requests.
        
        Args:
            requested_device: Optional specific device request (e.g., "cuda:1", "cpu", "mps")
            
        Returns:
            str: The device string to use
        """
        # If specific device requested, try to use it
        if requested_device is not None:
            try:
                # Validate the requested device is available
                if requested_device.startswith("cuda"):
                    if not torch.cuda.is_available():
                        logger.warning(f"Requested CUDA device {requested_device}, but CUDA is not available. Falling back to auto-detection.")
                    else:
                        device_idx = 0
                        if ":" in requested_device:
                            try:
                                device_idx = int(requested_device.split(":")[1])
                                if device_idx >= torch.cuda.device_count():
                                    logger.warning(f"Requested CUDA device {device_idx} out of range. Using device 0 instead.")
                                    device_idx = 0
                            except ValueError:
                                device_idx = 0
                        # Valid CUDA device
                        logger.info(f"Using requested CUDA device: cuda:{device_idx}")
                        return f"cuda:{device_idx}"
                elif requested_device == "mps":
                    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                        logger.info("Using requested MPS device (Apple Silicon)")
                        return "mps"
                    else:
                        logger.warning("MPS (Apple Silicon) was requested but is not available. Falling back to auto-detection.")
                elif requested_device == "cpu":
                    logger.info("Using requested CPU device")
                    return "cpu"
                else:
                    logger.warning(f"Unknown device requested: {requested_device}. Falling back to auto-detection.")
            except Exception as e:
                logger.warning(f"Error setting requested device {requested_device}: {e}. Falling back to auto-detection.")
        
        # Auto-detection logic
        if torch.cuda.is_available():
            # Get GPU memory info to detect if CUDA is usable
            try:
                # Get the device with most free memory
                device_count = torch.cuda.device_count()
                if device_count > 1:
                    # Multiple GPUs available, find the one with most free memory
                    free_memory = []
                    for i in range(device_count):
                        torch.cuda.set_device(i)
                        torch.cuda.empty_cache()
                        memory_reserved = torch.cuda.memory_reserved(i)
                        memory_allocated = torch.cuda.memory_allocated(i)
                        free = memory_reserved - memory_allocated
                        free_memory.append((i, free))
                    
                    # Sort by free memory (descending)
                    free_memory.sort(key=lambda x: x[1], reverse=True)
                    best_device = free_memory[0][0]
                    logger.info(f"Selected CUDA device {best_device} with most free memory")
                    return f"cuda:{best_device}"
                else:
                    # Only one GPU, use it
                    return "cuda:0"
            except Exception as e:
                logger.warning(f"Error determining best CUDA device: {e}. Using cuda:0.")
                return "cuda:0"
        
        # Check for Apple Silicon (MPS)
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            logger.info("CUDA not available. Using MPS device (Apple Silicon)")
            return "mps"
        
        # Fallback to CPU
        else:
            logger.info("No GPU detected. Using CPU")
            return "cpu"
    
    def _get_optimal_dtype(self):
        """
        Determine the optimal data type based on the device.
        
        Returns:
            torch.dtype: The optimal data type
        """
        # Apple Silicon MPS often works best with float32
        if self.device == "mps":
            logger.info("Using float32 for MPS device (Apple Silicon)")
            return torch.float32
            
        # For CUDA, try using float16 for better performance if supported
        elif self.device.startswith("cuda"):
            try:
                # Test if GPU supports float16 efficiently
                test_tensor = torch.zeros(1, device=self.device, dtype=torch.float16)
                # BF16 might be better on some GPUs, but requires newer hardware
                # Use it if available on newer cards
                if torch.cuda.get_device_capability(0)[0] >= 8:  # Ampere or newer
                    try:
                        test_bf16 = torch.zeros(1, device=self.device, dtype=torch.bfloat16)
                        logger.info("Using bfloat16 on modern GPU for better performance/accuracy balance")
                        return torch.bfloat16
                    except:
                        pass
                
                logger.info("Using float16 on GPU for better performance")
                return torch.float16
            except Exception as e:
                logger.warning(f"Float16 support issue: {e}. Falling back to float32.")
                return torch.float32
        
        # For CPU, always use float32
        else:
            return torch.float32
    
    def _log_device_info(self):
        """Log detailed device information for debugging"""
        try:
            if self.device.startswith("cuda"):
                gpu_idx = 0
                if ":" in self.device:
                    gpu_idx = int(self.device.split(":")[1])
                
                device_name = torch.cuda.get_device_name(gpu_idx)
                total_memory = torch.cuda.get_device_properties(gpu_idx).total_memory / 1e9  # Convert to GB
                
                logger.info(f"GPU: {device_name}")
                logger.info(f"Total memory: {total_memory:.2f} GB")
                logger.info(f"CUDA version: {torch.version.cuda}")
                
                # Log memory usage
                memory_allocated = torch.cuda.memory_allocated(gpu_idx) / 1e9  # GB
                memory_reserved = torch.cuda.memory_reserved(gpu_idx) / 1e9    # GB
                logger.info(f"Memory allocated: {memory_allocated:.2f} GB")
                logger.info(f"Memory reserved: {memory_reserved:.2f} GB")
                
            elif self.device == "mps":
                logger.info("Using Apple MPS device (Metal Performance Shaders)")
                import platform
                logger.info(f"Device: {platform.processor()}")
                
            else:  # CPU
                import platform
                logger.info(f"CPU: {platform.processor()}")
                
                # Try to get more detailed CPU info on Linux
                try:
                    if platform.system() == "Linux":
                        with open("/proc/cpuinfo", "r") as f:
                            for line in f:
                                if "model name" in line:
                                    logger.info(f"CPU Model: {line.split(':')[1].strip()}")
                                    break
                except:
                    pass
                
        except Exception as e:
            logger.debug(f"Error getting device info: {e}")
            # Non-critical, so just log and continue

    @abstractmethod
    def _setup_model(self) -> None:
        pass

    @abstractmethod
    def analyze_image(self, image_path: str, quality: str = "standard") -> Tuple[str, Optional[str]]:
        """
        Analyze an image using the model.
        
        Args:
            image_path (str): Path to the image file
            quality (str): Quality level - "standard", "detailed", or "creative"
            
        Returns:
            Tuple[str, Optional[str]]: (description, clean_caption)
        """
        raise NotImplementedError("Subclasses must implement analyze_image")

    @abstractmethod
    def analyze_images_batch(self, image_paths: list[str], quality: str = "standard") -> list[Tuple[str, Optional[str]]]:
        """
        Analyze a batch of images using the model.
        
        Args:
            image_paths (List[str]): List of paths to image files
            quality (str): Quality level - "standard", "detailed", or "creative"
            
        Returns:
            List[Tuple[str, Optional[str]]]: List of (description, clean_caption) tuples
        """
        raise NotImplementedError("Subclasses must implement analyze_images_batch")

    def clean_output(self, text: str) -> str:
        """Clean model output by removing special tokens, non-printable characters, and formatting."""
        import re
        try:
            # Ensure text is a string
            if not isinstance(text, str):
                try:
                    # Attempt to decode if it's bytes
                    if isinstance(text, bytes):
                        text = text.decode('utf-8', errors='replace')
                    else:
                        # If it's some other type, try to convert to string
                        text = str(text)
                except Exception as e_conv:
                    logger.error(f"Could not convert/decode text to string: {e_conv}")
                    return "Error: Could not decode model output."

            # Remove common special tokens
            text = text.replace('</s>', '').replace('<s>', '')
            text = re.sub(r'<\|.*?\|>', '', text) # For tokens like <|user|>, <|assistant|>
            text = re.sub(r'<image_placeholder>', '', text) # Specific to some models
            text = re.sub(r'<OD>', '', text) # Specific to Florence-2 OD task
            text = re.sub(r'<loc_\d+>', '', text) # Florence-2 location tokens

            # Remove any remaining HTML-like tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Remove non-printable characters except for common whitespace (space, tab, newline, carriage return)
            # This will also replace unknown/corrupted characters with the Unicode replacement character (U+FFFD)
            # if they couldn't be decoded properly earlier.
            text = ''.join(char if char.isprintable() or char in [' ', '\t', '\n', '\r'] else ' ' for char in text)

            # Normalize whitespace (replace multiple spaces/newlines with a single one)
            text = ' '.join(text.split())
            
            # Remove URLs
            text = re.sub(r'http\S+', '', text)
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error cleaning output text: {str(e)}")
            # If cleaning fails catastrophically, return a placeholder or the (potentially problematic) original text
            return "Error: Text cleaning failed." if not isinstance(text, str) else text
            
    def manage_memory(self, aggressive=False):
        """
        Free up memory for large model loading.
        
        Args:
            aggressive (bool): If True, uses more aggressive memory cleanup methods
                               that may impact performance but free more memory
        """
        try:
            # Clear CUDA cache if using CUDA
            if self.device.startswith("cuda"):
                logger.info("Clearing CUDA cache")
                torch.cuda.empty_cache()
                
                if aggressive:
                    # More aggressive memory cleanup for CUDA
                    if hasattr(torch.cuda, 'memory_stats'):
                        before = torch.cuda.memory_allocated() / 1e9
                        # Try to force garbage collection of CUDA tensors
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        after = torch.cuda.memory_allocated() / 1e9
                        logger.info(f"Aggressive cleanup: freed {before - after:.2f} GB")
            
            # For Apple Silicon MPS
            elif self.device == "mps" and hasattr(torch.mps, 'empty_cache'):
                logger.info("Clearing MPS cache")
                torch.mps.empty_cache()
            
            # For all devices, run Python's garbage collector
            import gc
            gc.collect()
            
            # Log memory info after cleanup
            if self.device.startswith("cuda"):
                device_idx = 0
                if ":" in self.device:
                    device_idx = int(self.device.split(":")[1])
                memory_allocated = torch.cuda.memory_allocated(device_idx) / 1e9
                logger.info(f"Memory allocated after cleanup: {memory_allocated:.2f} GB")
                
            return True
        except Exception as e:
            logger.warning(f"Error during memory management: {e}")
            return False
    
    def get_low_memory_optimization_settings(self):
        """
        Returns settings dictionary for low memory situations.
        
        These settings can be passed to model loading functions to reduce
        memory requirements, at the cost of some performance.
        
        Returns:
            dict: Settings to apply for low memory optimization
        """
        settings = {
            "low_cpu_mem_usage": True,
            "offload_folder": "offload_folder"  # For offloading to disk
        }
        
        # CPU-specific optimizations
        if self.device == "cpu":
            settings.update({
                "use_safetensors": True,  # More memory efficient loading format
                "torch_dtype": torch.float32,  # Use full precision on CPU
                "offload_state_dict": True,  # Offload weights when possible
            })
        
        # CUDA optimizations for low memory
        elif self.device.startswith("cuda"):
            # Get device capabilities
            device_idx = 0
            if ":" in self.device:
                device_idx = int(self.device.split(":")[1])
                
            # Check available GPU memory
            try:
                total_memory = torch.cuda.get_device_properties(device_idx).total_memory / 1e9
                if total_memory < 6:  # Less than 6GB VRAM
                    logger.info(f"Low VRAM detected ({total_memory:.1f}GB). Using 8-bit quantization.")
                    # Use 8-bit quantization for very low memory
                    settings.update({
                        "load_in_8bit": True,
                        "device_map": "auto",
                    })
                elif total_memory < 12:  # Less than 12GB VRAM
                    # Use 4-bit quantization if appropriate
                    logger.info(f"Medium VRAM detected ({total_memory:.1f}GB). Using 4-bit quantization.")
                    settings.update({
                        "load_in_4bit": True, 
                        "bnb_4bit_compute_dtype": torch.float16,
                        "device_map": "auto",
                    })
                else:
                    # For high memory GPUs, just use float16
                    settings.update({
                        "torch_dtype": torch.float16,
                        "device_map": "auto",
                    })
            except Exception as e:
                logger.warning(f"Error determining GPU memory: {e}")
                # Fallback to safe settings
                settings.update({
                    "device_map": "auto",
                    "torch_dtype": torch.float16
                })
        
        # Apple Silicon MPS
        elif self.device == "mps":
            settings.update({
                "torch_dtype": torch.float32,  # MPS works better with float32
                "use_safetensors": True
            })
            
        return settings
