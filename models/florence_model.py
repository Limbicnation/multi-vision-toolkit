# models/florence_model.py
from models.base_model import BaseVisionModel
from typing import Tuple, Optional, List
import logging
import torch
import importlib
from PIL import Image
import os
import sys
import platform

# Handle imports in a way that prevents circular dependencies
# Import only what's needed initially, defer other imports
import importlib
import logging

# Safely check for torch/torchvision versions without direct imports
def _check_installed_version(package_name):
    try:
        if importlib.util.find_spec(package_name) is None:
            return None
        
        package = importlib.import_module(package_name)
        if hasattr(package, '__version__'):
            return package.__version__
        return "Unknown version"
    except Exception as e:
        logging.warning(f"Error checking {package_name} version: {str(e)}")
        return None

# Check versions without full imports
torch_version = _check_installed_version('torch')
torchvision_version = _check_installed_version('torchvision')

# Log basic version info
logging.info(f"PyTorch version: {torch_version}")
logging.info(f"Torchvision version: {torchvision_version}")

# Import torch, but defer torchvision import to when it's actually needed
try:
    import torch
except ImportError as e:
    logging.error(f"PyTorch import error: {str(e)}")
    logging.error("PyTorch is required for all models. Please install it first.")
    # We don't raise here, as the error will be handled in dependency check

logger = logging.getLogger(__name__)

class Florence2Model(BaseVisionModel):
    """Florence-2 vision model implementation following official guidelines."""
    
    REQUIRED_PACKAGES = {
        'transformers': 'transformers',
        'timm': 'timm',
        'einops': 'einops',
        'torch': 'torch',
        'torchvision': 'torchvision>=0.17.0',
        'PIL': 'Pillow'
    }

    def __init__(self):
        """Initialize Florence-2 model with dependency checks."""
        self._check_dependencies()
        super().__init__()

    @classmethod
    def _check_dependencies(cls) -> None:
        """Check for required dependencies and provide clear installation instructions."""
        missing_packages = []
        for package, pip_name in cls.REQUIRED_PACKAGES.items():
            try:
                # Special handling for torchvision due to potential circular import
                if package == 'torchvision':
                    try:
                        import torchvision
                        # Verify we can access a specific attribute to confirm proper import
                        # This will catch circular import issues
                        version = torchvision.__version__
                    except (ImportError, AttributeError) as e:
                        logger.warning(f"Issue with torchvision: {str(e)}")
                        missing_packages.append((package, pip_name))
                else:
                    importlib.import_module(package)
            except ImportError:
                missing_packages.append((package, pip_name))

        if missing_packages:
            install_command = "pip install " + " ".join(pkg[1] for pkg in missing_packages)
            error_msg = (
                f"Missing required packages: {', '.join(pkg[0] for pkg in missing_packages)}\n"
                f"Please install them using:\n{install_command}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _setup_model(self) -> None:
        """Set up the Florence-2 model following official implementation."""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            import torch
            
            logger.info("Loading Florence-2 model...")
            # Try local path first, then fall back to HuggingFace
            local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "weights", "Florence-2-base")
            if os.path.exists(local_path):
                model_path = local_path
                logger.info(f"Using local model: {model_path}")
            else:
                model_path = "microsoft/Florence-2-base"
                logger.info(f"Using remote model: {model_path}")
            
            # Check PyTorch version for vulnerability warning
            torch_version = torch.__version__
            required_version = "2.6.0"
            required_torchvision = "0.17.0"
            
            # Try to parse versions for comparison
            try:
                torch_ver_parts = torch_version.split('.')
                required_ver_parts = required_version.split('.')
                
                version_too_old = False
                for i in range(min(len(torch_ver_parts), len(required_ver_parts))):
                    if int(torch_ver_parts[i]) < int(required_ver_parts[i]):
                        version_too_old = True
                        break
                    elif int(torch_ver_parts[i]) > int(required_ver_parts[i]):
                        break
                
                if version_too_old:
                    logger.warning(f"Your PyTorch version ({torch_version}) is older than the recommended minimum ({required_version})")
                    logger.warning("You may encounter security warnings or errors with torch.load due to CVE-2025-32434")
                    
                    # Provide platform-specific update instructions
                    if sys.platform == "win32":
                        logger.warning("To update on Windows, run:")
                        logger.warning("pip install torch==2.6.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121")
                    elif sys.platform == "linux":
                        if "microsoft" in os.uname().release:  # WSL detection
                            logger.warning("Windows WSL detected. To update, run:")
                            logger.warning("pip install torch==2.6.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121")
                        else:
                            logger.warning("To update on Linux, run:")
                            logger.warning("pip install torch==2.6.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121")
                    elif sys.platform == "darwin":
                        # Apple Silicon vs Intel Mac
                        import platform
                        if platform.processor() == "arm":
                            logger.warning("To update on Apple Silicon Mac, run:")
                            logger.warning("pip install torch==2.6.0 torchvision==0.17.0")
                        else:
                            logger.warning("To update on Intel Mac, run:")
                            logger.warning("pip install torch==2.6.0 torchvision==0.17.0")
                    else:
                        logger.warning("Consider upgrading: pip install torch>=2.6.0 torchvision>=0.17.0")
            except Exception as e:
                # If parsing failed, log it but continue
                logger.debug(f"Failed to parse PyTorch version: {str(e)}")
            
            # Initialize model with added error handling for dependency issues
            try:
                # Check if torchvision is properly imported
                if 'torchvision' not in sys.modules:
                    logger.info("Importing torchvision at model initialization time")
                    try:
                        import torchvision
                    except Exception as tv_error:
                        logger.warning(f"Torchvision import error (non-critical): {str(tv_error)}")
                
                # Handle timm import issues which are common with Florence models
                try:
                    # Import timm explicitly if not already imported
                    if 'timm' not in sys.modules:
                        logger.info("Importing timm explicitly")
                        import timm
                    
                    # Original critical fix - this is what actually makes Florence-2 work
                    if 'timm.models.layers' in sys.modules:
                        logger.warning("Detected potential problematic timm import. Attempting workaround...")
                        import timm.layers
                        # This is the key line - directly replace the module in sys.modules
                        sys.modules['timm.models.layers'] = timm.layers
                    
                    # Additional fixes for different timm versions
                    elif 'timm.models.layers' in sys.modules and 'timm.layers' in sys.modules:
                        logger.info("Detected both timm modules. Applying circular import workaround...")
                        sys.modules['timm.models.layers'] = sys.modules['timm.layers']
                    
                    # Alternative import approach for newer timm versions
                    elif 'timm.models.layers' not in sys.modules:
                        logger.info("Using alternative timm import pattern")
                        try:
                            import timm.models.layers
                        except ImportError:
                            # If that direct import fails, try creating it
                            import timm.layers
                            sys.modules['timm.models.layers'] = timm.layers
                except Exception as timm_error:
                    logger.warning(f"Timm import fix attempts failed (non-critical): {str(timm_error)}")
                    logger.info("Model may still load successfully despite timm import issues")
                
                logger.info("Attempting to load model with safetensors...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=True,
                    use_safetensors=True,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    revision="main",  # Explicitly use main branch
                    local_files_only=False  # Allow downloading if not available locally
                )
            except Exception as e:
                logger.warning(f"Failed to load with safetensors: {str(e)}")
                logger.info("Attempting to load with standard method...")
                
                try:
                    # Fallback to standard loading with explicit device
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=self.torch_dtype,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        local_files_only=False,  # Force download if needed
                        revision="main"  # Explicitly use main branch
                    ).to(self.device)
                except Exception as load_error:
                    logger.error(f"Failed to load model: {str(load_error)}")
                    
                    # Provide detailed installation instructions based on common error types
                    error_str = str(load_error).lower()
                    
                    if "safetensors" in error_str:
                        extra_package = "pip install safetensors"
                    elif "timm" in error_str:
                        extra_package = "pip install timm==0.9.12"
                    elif "einops" in error_str:
                        extra_package = "pip install einops"
                    else:
                        extra_package = "pip install timm==0.9.12 safetensors einops"
                    
                    # Platform-specific installation commands
                    if sys.platform == "win32" or ("microsoft" in os.uname().release and sys.platform == "linux"):
                        # Windows or WSL
                        torch_install = "pip install torch==2.6.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121"
                    elif sys.platform == "darwin" and platform.processor() == "arm":
                        # Apple Silicon
                        torch_install = "pip install torch==2.6.0 torchvision==0.17.0"
                    else:
                        # Linux or other
                        torch_install = "pip install torch==2.6.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121"
                    
                    error_message = (
                        f"Model loading failed. This may be due to a PyTorch security restriction or dependency issue.\n\n"
                        f"Try installing the exact required versions:\n"
                        f"1. {torch_install}\n"
                        f"2. {extra_package}\n"
                        f"3. pip install transformers accelerate\n\n"
                        f"Original error: {str(load_error)}"
                    )
                    raise RuntimeError("Model initialization failed") from load_error

            # Load processor (less likely to have issues)
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=False  # Allow downloading if not available locally
                )
            except Exception as e:
                logger.error(f"Failed to load processor: {str(e)}")
                raise RuntimeError("Processor initialization failed") from e

        except Exception as e:
            error_msg = f"Failed to load Florence-2 model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def analyze_image(self, image_path: str, quality: str = "standard") -> Tuple[str, Optional[str]]: # Add quality parameter
        """Analyze an image using the Florence-2 model."""
        # Florence-2 doesn't inherently have quality levels like "detailed" or "creative"
        # in the same way as some generative models. Its tasks are more specific.
        # You can choose to ignore the 'quality' param or perhaps adapt the prompt
        # slightly if you find a way, but for now, just accepting it fixes the TypeError.
        logger.info(f"Florence2Model received quality='{quality}', but it's not currently used for generation adjustments.")
        try:
            # Validate image path
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return "Error: Image file not found.", None

            # Load and validate image
            try:
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {str(e)}")
                return "Error: Failed to load or process image.", None

            # Generate detailed caption
            try:
                # Caption generation
                caption_inputs = self.processor(
                    text="<image>Describe this image in detail:",
                    images=image,
                    return_tensors="pt"
                ).to(self.device, self.torch_dtype)

                with torch.inference_mode():
                    caption_ids = self.model.generate(
                        input_ids=caption_inputs["input_ids"],
                        pixel_values=caption_inputs["pixel_values"],
                        max_new_tokens=256,
                        num_beams=3,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                
                caption = self.processor.batch_decode(
                    caption_ids,
                    skip_special_tokens=True
                )[0]

                # Object detection
                od_inputs = self.processor(
                    text="<OD>",
                    images=image,
                    return_tensors="pt"
                ).to(self.device, self.torch_dtype)

                with torch.inference_mode():
                    od_ids = self.model.generate(
                        input_ids=od_inputs["input_ids"],
                        pixel_values=od_inputs["pixel_values"],
                        max_new_tokens=128,
                        num_beams=3,
                        do_sample=False
                    )
                
                od_text = self.processor.batch_decode(od_ids, skip_special_tokens=False)[0]
                objects = self.processor.post_process_generation(
                    od_text,
                    task="<OD>",
                    image_size=(image.width, image.height)
                )

            except Exception as e:
                logger.error(f"Error generating analysis: {str(e)}")
                return "Error: Failed to generate image analysis.", None

            # Process and return results
            try:
                caption = self.clean_output(caption)
                objects_str = ", ".join(str(obj) for obj in objects) if objects else ""
                
                description = f"Description: {caption}"
                if objects_str:
                    description += f"\n\nDetected objects: {objects_str}"
                
                return description, caption
            except Exception as e:
                logger.error(f"Error processing model output: {str(e)}")
                return "Error: Failed to process model output.", None

        except Exception as e:
            logger.error(f"Error analyzing image with Florence-2: {str(e)}")
            return "Error: An unexpected error occurred.", None

    @classmethod
    def is_available(cls) -> bool:
        """Check if the model can be initialized with current environment."""
        try:
            # Check for required dependencies except timm which might have issues
            import torch
            import transformers
            
            # Test a direct import from transformers for the necessary classes
            try:
                from transformers import AutoProcessor, AutoModelForCausalLM
                has_transformers = True
            except ImportError:
                logger.error("Required transformers classes not found. Please update transformers.")
                has_transformers = False
                
            # Check CUDA availability
            has_cuda = torch.cuda.is_available()
            if not has_cuda:
                logger.warning("CUDA not available. Model will run in CPU mode with greatly reduced performance.")
                
            return has_transformers
        except Exception as e:
            logger.error(f"Failed to check model availability: {str(e)}")
            return False
