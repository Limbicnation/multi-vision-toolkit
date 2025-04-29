# models/florence_model.py
from models.base_model import BaseVisionModel
from typing import Tuple, Optional, List
import logging
import torch
import importlib
from PIL import Image
import os
import sys

# Handle potential circular import with torchvision
try:
    # Try to import torchvision directly
    import torchvision
except (ImportError, AttributeError) as e:
    logging.warning(f"Issue with torchvision import: {str(e)}")
    # If there's a circular import issue, we'll fix it later in the model setup

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
            model_path = "microsoft/Florence-2-large"
            
            # Check PyTorch version for vulnerability warning
            torch_version = torch.__version__
            required_version = "2.6.0"
            
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
                    logger.warning("You may encounter security warnings or errors with torch.load")
                    logger.warning("Consider upgrading: pip install torch>=2.6.0 torchvision>=0.17.0")
            except Exception:
                # If parsing failed, just continue
                pass
            
            # Initialize model with added error handling for timm issues
            try:
                # Replace timm import in globals to prevent circular imports later
                try:
                    import sys
                    if 'timm.models.layers' in sys.modules:
                        logger.warning("Detected potential problematic timm import. Attempting workaround...")
                        import timm.layers
                        sys.modules['timm.models.layers'] = timm.layers
                except Exception as timm_error:
                    logger.warning(f"Timm import fix failed (non-critical): {str(timm_error)}")
                
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
                    error_message = (
                        f"Model loading failed. This may be due to a PyTorch security restriction or timm compatibility issue.\n"
                        f"Try installing the exact required versions:\n"
                        f"pip install torch==2.6.0 torchvision==0.17.0 timm==0.9.12\n\n"
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

    def analyze_image(self, image_path: str) -> Tuple[str, Optional[str]]:
        """Analyze an image using the Florence-2 model."""
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