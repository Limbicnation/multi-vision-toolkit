# models/janus_model.py
from models.base_model import BaseVisionModel
import logging
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from typing import Tuple, Optional
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class JanusModel(BaseVisionModel):
    """
    Vision model implementation using BLIP model for image understanding.
    This is a publicly available model that doesn't require authentication.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the model.
        
        Args:
            model_path (str): HuggingFace model path
        """
        # Try local path first, then fall back to HuggingFace
        if model_path is None:
            local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "weights", "blip-image-captioning-base")
            if os.path.exists(local_path):
                self.model_path = local_path
                logger.info(f"Using local model: {self.model_path}")
            else:
                self.model_path = "Salesforce/blip-image-captioning-base"
                logger.info(f"Using remote model: {self.model_path}")
        else:
            self.model_path = model_path
            logger.info(f"Using specified model: {self.model_path}")
        
        super().__init__()

    def _setup_model(self) -> None:
        """Set up the BLIP model."""
        try:
            logger.info(f"Loading model from {self.model_path}...")
            
            try:
                self.processor = BlipProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,  # Add trust_remote_code
                    local_files_only=False  # Allow downloading if not available locally
                )
            except Exception as e:
                logger.error(f"Failed to load processor: {str(e)}")
                raise RuntimeError("Processor initialization failed") from e
                
            try:
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=True,  # Add trust_remote_code
                    revision="main",         # Explicitly use main branch
                    local_files_only=False   # Allow downloading if not available locally
                ).to(self.device)
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise RuntimeError("Model initialization failed") from e
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def analyze_image(self, image_path: str, quality: str = "standard") -> Tuple[str, Optional[str]]:
        """
        Analyze an image using the BLIP model.
        
        Args:
            image_path (str): Path to the image file
            quality (str): Quality level - "standard", "detailed", or "creative"
            
        Returns:
            Tuple[str, Optional[str]]: (description, clean_caption)
        """
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
                
            # Generate caption
            try:
                # Prepare inputs
                inputs = self.processor(
                    images=image, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Set generation parameters based on quality
                if quality == "detailed":
                    max_length = 75
                    num_beams = 7
                    temperature = 0.65
                    top_p = 0.95
                    length_penalty = 1.2
                elif quality == "creative":
                    max_length = 75
                    num_beams = 5
                    temperature = 1.0
                    top_p = 0.95
                    length_penalty = 0.8
                    do_sample = True
                else:  # standard
                    max_length = 50
                    num_beams = 5
                    temperature = 0.7
                    top_p = 0.9
                    length_penalty = 1.0
                    do_sample = True
                
                # Generate caption
                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        pixel_values=inputs.pixel_values,
                        max_length=max_length,
                        num_beams=num_beams,
                        length_penalty=length_penalty,
                        temperature=temperature,
                        do_sample=do_sample,
                        top_p=top_p
                    )
                
                # Decode caption
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                caption = self.clean_output(caption)
                
                # Prepare detailed description
                description = f"Description: {caption}"
                
                return description, caption
                
            except Exception as e:
                logger.error(f"Error generating caption: {str(e)}")
                return "Error: Failed to generate image description.", None
                
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return "Error: An unexpected error occurred.", None

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if the model can be initialized with current environment.
        """
        try:
            import torch
            import transformers
            
            # Check CUDA availability
            has_cuda = torch.cuda.is_available()
            if not has_cuda:
                logger.warning("CUDA not available. Model will run in CPU mode with reduced performance.")
            
            return True
        except ImportError as e:
            logger.error(f"Required package not found: {str(e)}")
            return False