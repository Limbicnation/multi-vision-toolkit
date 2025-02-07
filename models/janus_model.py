# models/janus_model.py
from models.base_model import BaseVisionModel
import logging
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from typing import Tuple, Optional
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class JanusModel(BaseVisionModel):
    """Janus vision model implementation with improved error handling and configuration."""
    
    def __init__(self, model_path: str = "janhq/janus-1-7b"):
        """
        Initialize the Janus model.
        
        Args:
            model_path (str): HuggingFace model path
        """
        self.model_path = model_path
        super().__init__()

    def _load_token(self) -> Optional[str]:
        """
        Load HuggingFace token from environment with fallback mechanisms.
        """
        try:
            # Try loading from environment first
            token = os.getenv('HF_TOKEN')
            if token:
                return token
                
            # Try loading from .env file if python-dotenv is available
            try:
                from dotenv import load_dotenv
                env_path = Path('.env')
                if env_path.exists():
                    load_dotenv()
                    token = os.getenv('HF_TOKEN')
                    if token:
                        return token
            except ImportError:
                logger.warning("python-dotenv not installed. Skipping .env file loading.")
                
            logger.warning("No HF_TOKEN found in environment or .env file")
            return None
            
        except Exception as e:
            logger.error(f"Error loading token: {str(e)}")
            return None

    def _setup_model(self) -> None:
        """
        Set up the Janus model with improved error handling and configuration.
        """
        try:
            logger.info(f"Loading Janus model from {self.model_path}...")
            
            # Load token
            token = self._load_token()
            
            # Model configuration with safe defaults
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
                "device_map": "balanced",
                "trust_remote_code": True,
            }
            
            # Only enable 8-bit loading if CUDA is available
            if torch.cuda.is_available():
                model_kwargs["load_in_8bit"] = True
            
            # Add token if available
            if token:
                model_kwargs["token"] = token
            
            # Initialize model and processor
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise RuntimeError("Model initialization failed") from e
                
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    token=token if token else None
                )
            except Exception as e:
                logger.error(f"Failed to load processor: {str(e)}")
                raise RuntimeError("Processor initialization failed") from e
                
        except Exception as e:
            logger.error(f"Failed to initialize Janus model: {str(e)}")
            raise

    def analyze_image(self, image_path: str) -> Tuple[str, Optional[str]]:
        """
        Analyze an image using the Janus model.
        
        Args:
            image_path (str): Path to the image file
            
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
                
            # Prepare model inputs
            try:
                inputs = self.processor(
                    images=image,
                    text="Describe this image in detail.",
                    return_tensors="pt"
                ).to(self.device)
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return "Error: Failed to process image for model input.", None
            
            # Generate caption
            try:
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        num_beams=5,
                        length_penalty=1.0,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        use_cache=True,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id
                    )
            except Exception as e:
                logger.error(f"Error generating caption: {str(e)}")
                return "Error: Failed to generate image description.", None
            
            # Process output
            try:
                caption = self.processor.decode(outputs[0], skip_special_tokens=True)
                caption = self.clean_output(caption)
                description = f"Description: {caption}"
                return description, caption
            except Exception as e:
                logger.error(f"Error processing model output: {str(e)}")
                return "Error: Failed to process model output.", None
                
        except Exception as e:
            logger.error(f"Error analyzing image with Janus model: {str(e)}")
            return "Error: An unexpected error occurred.", None

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if the model can be initialized with current environment.
        
        Returns:
            bool: True if model can be initialized, False otherwise
        """
        try:
            # Check for required packages
            import torch
            import transformers
            
            # Check CUDA availability for 8-bit loading
            has_cuda = torch.cuda.is_available()
            if not has_cuda:
                logger.warning("CUDA not available. Model will run in CPU mode with reduced performance.")
            
            return True
        except ImportError as e:
            logger.error(f"Required package not found: {str(e)}")
            return False