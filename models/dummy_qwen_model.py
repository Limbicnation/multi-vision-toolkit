# models/dummy_qwen_model.py
from models.base_model import BaseVisionModel
import logging
from typing import Tuple, Optional
import os

logger = logging.getLogger(__name__)

class QwenModel(BaseVisionModel):
    """Dummy Qwen model for when the actual model cannot be loaded."""
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"):
        """Initialize the dummy Qwen model."""
        self.model_path = model_path
        logger.warning("Using dummy QwenModel - actual model could not be loaded")
        # Skip dependency checks but call parent init
        super().__init__()
    
    def _setup_model(self) -> None:
        """Setup dummy model."""
        logger.warning("Using dummy QwenModel with no actual model loaded")
        self.model = None
        self.processor = None
    
    def analyze_image(self, image_path: str, quality: str = "standard") -> Tuple[str, Optional[str]]:
        """Return a dummy analysis result with fallback to CLIP model."""
        if not os.path.exists(image_path):
            return "Error: Image file not found.", None
            
        # First try to use a fallback CLIP model which is widely available
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            logger.info("Using CLIP model as a Qwen model fallback")
            
            # Configure cache
            cache_options = {}
            if "TRANSFORMERS_CACHE" in os.environ:
                cache_options["cache_dir"] = os.environ["TRANSFORMERS_CACHE"]
                
            # Set device and dtype
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # Load model with proper device configuration
            model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", 
                torch_dtype=dtype,
                **cache_options
            )
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", **cache_options)
            
            # Load and process image
            from PIL import Image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Get device
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            # Move model to the same device
            model = model.to(device)
            
            # Prepare inputs
            inputs = processor(
                text=["a photo of a landscape", "a portrait", "a photo of food", 
                      "a photo of an animal", "a photo of a building", "a photo of people"],
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to the same device as model and cast to model's dtype
            for key in inputs:
                if hasattr(inputs[key], "to"):
                    inputs[key] = inputs[key].to(device)
                    # Only cast floating-point tensors to the model's dtype
                    if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                        inputs[key] = inputs[key].to(dtype)
            
            # Get prediction
            with torch.inference_mode():
                outputs = model(**inputs)
                
            # Get the image-text similarity scores
            logits_per_image = outputs.logits_per_image
            
            # Move to CPU before converting to Python objects
            if device != "cpu":
                logits_per_image = logits_per_image.cpu()
                
            probs = logits_per_image.softmax(dim=1).tolist()[0]
            
            # Get the most likely category
            categories = ["landscape", "portrait", "food", "animal", "building", "people"]
            scores = list(zip(categories, probs))
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Generate appropriate caption based on quality
            top_category = scores[0][0]
            confidence = scores[0][1] * 100
            
            if quality == "detailed":
                # More detailed description
                other_elements = [f"{cat} ({conf:.1f}%)" for cat, conf in scores[1:3]]
                caption = f"This image appears to be a {top_category} (confidence: {confidence:.1f}%). "
                caption += f"It may also contain elements of {' and '.join(other_elements)}."
            elif quality == "creative":
                # More creative description
                caption = f"An interesting {top_category} scene captured in this image. "
                if top_category == "landscape":
                    caption += "The natural beauty unfolds across the frame, inviting the viewer to explore its details."
                elif top_category == "portrait":
                    caption += "The subject's presence dominates the composition, telling a story through expression and posture."
                elif top_category == "food":
                    caption += "The culinary delight is presented with care, enticing the viewer's appetite."
                elif top_category == "animal":
                    caption += "The creature's character and form create a compelling focal point in the image."
                elif top_category == "building":
                    caption += "The architectural elements reveal both function and aesthetic considerations."
                else:  # people
                    caption += "The human elements bring life and scale to the composition."
            else:  # standard
                caption = f"This image shows a {top_category}."
                
            description = f"Description: {caption}\n\nGenerated by: CLIP (Fallback Mode)"
            return description, caption
            
        except Exception as e:
            # If CLIP fails, show the original error message
            logger.error(f"CLIP fallback failed: {str(e)}")
            message = (
                "Error: Qwen model is not available due to import errors.\n\n"
                "Try installing the required dependencies:\n"
                "- pip install torch==2.6.0 torchvision==0.17.0\n"
                "- pip install transformers==4.36.2\n"
                "- pip install qwen-vl-utils[decord]==0.0.8\n"
                "- pip install accelerate einops timm==0.9.2\n"
                "- pip install ftfy regex tqdm safetensors\n\n"
                "A fallback CLIP model was attempted but also failed.\n"
                "Or try using another model like 'janus'"
            )
            return message, None
    
    @classmethod
    def is_available(cls) -> bool:
        """This dummy model is always available as a fallback."""
        return True
