# models/dummy_janus_model.py
from models.base_model import BaseVisionModel
import logging
from typing import Tuple, Optional
import os

logger = logging.getLogger(__name__)

class JanusModel(BaseVisionModel):
    """Dummy Janus model for when the actual model cannot be loaded."""
    
    def __init__(self, model_path: str = "Salesforce/blip-image-captioning-large"):
        """Initialize the dummy Janus model."""
        self.model_path = model_path
        logger.warning("Using dummy JanusModel - actual model could not be loaded")
        # Skip dependency checks but call parent init
        super().__init__()
    
    def _setup_model(self) -> None:
        """Setup dummy model."""
        logger.warning("Using dummy JanusModel with no actual model loaded")
        self.model = None
        self.processor = None
    
    def analyze_image(self, image_path: str, quality: str = "standard") -> Tuple[str, Optional[str]]:
        """Return a dummy analysis result."""
        if not os.path.exists(image_path):
            return "Error: Image file not found.", None
            
        message = (
            "Error: Janus model (BLIP) is not available due to import errors.\n\n"
            "Try installing the required dependencies:\n"
            "- pip install torch==2.6.0 torchvision==0.17.0\n"
            "- pip install transformers>=4.30.0\n"
            "- pip install accelerate\n\n"
            "Or continue using the Florence-2 model which is working correctly."
        )
        
        return message, None
    
    @classmethod
    def is_available(cls) -> bool:
        """This dummy model is always available as a fallback."""
        return True