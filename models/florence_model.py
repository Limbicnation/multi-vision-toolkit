# models/florence_model.py
from models.base_model import BaseVisionModel
from typing import Tuple, Optional, List
import logging
import torch
import importlib
from PIL import Image
import os

logger = logging.getLogger(__name__)

class Florence2Model(BaseVisionModel):
    """Florence-2 vision model implementation following official guidelines."""
    
    REQUIRED_PACKAGES = {
        'transformers': 'transformers',
        'timm': 'timm',
        'einops': 'einops',
        'torch': 'torch',
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
            
            logger.info("Loading Florence-2 model...")
            model_path = "microsoft/Florence-2-large"
            
            # Initialize model without device_map for proper loading
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=True
                ).to(self.device)  # Move to device after initialization
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise RuntimeError("Model initialization failed") from e

            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True
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
            cls._check_dependencies()
            return True
        except Exception:
            return False