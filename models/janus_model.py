# models/janus_model.py
from models.base_model import BaseVisionModel
import logging
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from typing import Tuple, Optional, List
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

    def get_prompt_for_quality(self, quality: str = "standard") -> str:
        """
        Get the appropriate prompt text for Janus/BLIP model based on quality.
        
        Args:
            quality: Quality level - "standard", "detailed", or "creative"
            
        Returns:
            str: Prompt text optimized for the quality mode
        """
        if quality == "detailed":
            return "Provide a comprehensive and detailed description of this image. Include all visible elements, colors, positioning, and background details."
        elif quality == "creative":
            return "Create an imaginative and evocative description of this image. Use vivid language, metaphors, and creative interpretations."
        else:  # standard
            return "Describe this image in a concise way."
    
    def get_generation_params_for_quality(self, quality: str = "standard") -> dict:
        """
        Get the appropriate generation parameters for Janus/BLIP model based on quality.
        
        Args:
            quality: Quality level - "standard", "detailed", or "creative"
            
        Returns:
            dict: Generation parameters optimized for the quality mode
        """
        if quality == "detailed":
            return {
                "max_length": 100,       # Longer for detailed descriptions
                "min_length": 60,        # Ensure a minimum length
                "num_beams": 7,          # More beams for higher quality
                "temperature": 0.65,     # Lower temperature for factual output
                "top_p": 0.92,           # Slightly restrictive filtering
                "length_penalty": 1.5,   # Encourage longer outputs
                "do_sample": False,      # Deterministic for factual details
                "repetition_penalty": 1.2 # Avoid repetition
            }
        elif quality == "creative":
            return {
                "max_length": 90,        # Medium length
                "min_length": 30,        # Lower minimum for more flexibility
                "num_beams": 5,          # Fewer beams for more variety
                "temperature": 1.1,      # Higher temperature for more creativity
                "top_p": 0.97,           # Less restrictive top_p
                "top_k": 60,             # Add top_k for diversity
                "length_penalty": 0.8,   # Shorter outputs acceptable
                "do_sample": True,       # Enable sampling for creativity
                "repetition_penalty": 1.0 # No repetition penalty to allow stylistic repetition
            }
        else:  # standard
            return {
                "max_length": 50,        # Short, concise outputs
                "min_length": 10,        # Very low minimum
                "num_beams": 5,          # Standard beam count
                "temperature": 0.7,      # Standard temperature
                "top_p": 0.9,            # Standard top_p
                "length_penalty": 1.0,   # Neutral length penalty
                "do_sample": True,       # Some randomness
                "repetition_penalty": 1.1 # Light repetition penalty
            }
    
    def analyze_image(self, image_path: str, quality: str = "standard") -> Tuple[str, Optional[str]]:
        """
        Analyze an image using the BLIP model with quality-specific settings.
        
        Args:
            image_path: Path to the image file
            quality: Quality level - "standard", "detailed", or "creative"
            
        Returns:
            Tuple[str, Optional[str]]: (description, clean_caption)
        """
        logger.info(f"JanusModel using quality mode: '{quality}'")
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
                
            # Generate caption with quality-specific settings
            try:
                # Get quality-specific prompt and generation parameters
                text_prompt = self.get_prompt_for_quality(quality)
                generation_params = self.get_generation_params_for_quality(quality)
                
                # Prepare inputs with the quality-specific prompt
                inputs = self.processor(
                    text=text_prompt,
                    images=image, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate caption
                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        pixel_values=inputs.pixel_values,
                        **generation_params
                    )
                
                # Decode caption
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                caption = self.clean_output(caption)
                
                # Format description based on quality
                if quality == "detailed":
                    description = f"Detailed Analysis: {caption}"
                    # Add image metadata for detailed view
                    width, height = image.size
                    description += f"\n\nImage Dimensions: {width}x{height} pixels"
                    description += f"\n\nDominant Colors: {self._analyze_colors(image)}"
                elif quality == "creative":
                    description = f"Creative Interpretation: {caption}"
                    # For creative mode, add a stylistic note
                    description += "\n\nThis creative description captures the essence and mood of the image through artistic interpretation."
                else:  # standard
                    description = f"Description: {caption}"
                
                description += f"\n\nGenerated using Janus/BLIP ({quality} mode)"
                
                return description, caption
                
            except Exception as e:
                logger.error(f"Error generating caption: {str(e)}")
                return "Error: Failed to generate image description.", None
                
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return "Error: An unexpected error occurred.", None
    
    def _analyze_colors(self, image: Image.Image) -> str:
        """
        Analyze the dominant colors in an image for the detailed mode.
        This is a simple implementation - returns basic color analysis.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            str: Description of dominant colors
        """
        try:
            # Resize image for faster processing
            small_img = image.resize((100, 100))
            # Convert to RGB if not already
            if small_img.mode != 'RGB':
                small_img = small_img.convert('RGB')
            
            # Simple brightness analysis
            r, g, b = 0, 0, 0
            pixel_count = 0
            
            for x in range(small_img.width):
                for y in range(small_img.height):
                    pr, pg, pb = small_img.getpixel((x, y))
                    r += pr
                    g += pg
                    b += pb
                    pixel_count += 1
            
            # Average RGB values
            r //= pixel_count
            g //= pixel_count
            b //= pixel_count
            
            # Very simple color categorization
            brightness = (r + g + b) // 3
            if brightness < 85:
                brightness_desc = "dark"
            elif brightness < 170:
                brightness_desc = "medium"
            else:
                brightness_desc = "bright"
            
            # Determine color balance
            max_channel = max(r, g, b)
            if max_channel == r and r > g + 20 and r > b + 20:
                color_desc = "reddish"
            elif max_channel == g and g > r + 20 and g > b + 20:
                color_desc = "greenish"
            elif max_channel == b and b > r + 20 and b > g + 20:
                color_desc = "bluish"
            elif r > 200 and g > 200 and b < 100:
                color_desc = "yellowish"
            elif r > 200 and g < 100 and b > 200:
                color_desc = "magenta"
            elif r < 100 and g > 200 and b > 200:
                color_desc = "cyan"
            elif abs(r - g) < 30 and abs(r - b) < 30 and abs(g - b) < 30:
                if brightness < 60:
                    color_desc = "black"
                elif brightness > 200:
                    color_desc = "white"
                else:
                    color_desc = "gray"
            else:
                color_desc = "mixed"
            
            return f"{brightness_desc} {color_desc} tones (RGB avg: {r},{g},{b})"
        except Exception as e:
            logger.warning(f"Color analysis error: {e}")
            return "color analysis unavailable"

    def analyze_images_batch(self, image_paths: List[str], quality: str = "standard") -> List[Tuple[str, Optional[str]]]:
        """
        Analyze a batch of images using the BLIP model by processing them individually.
        
        Args:
            image_paths (List[str]): List of paths to image files
            quality (str): Quality level - "standard", "detailed", or "creative"
            
        Returns:
            List[Tuple[str, Optional[str]]]: List of (description, clean_caption) tuples
        """
        if not image_paths:
            return []
        
        results = []
        for image_path in image_paths:
            # Call the existing single-image analysis method
            description, clean_caption = self.analyze_image(image_path, quality=quality)
            results.append((description, clean_caption))
        return results

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
