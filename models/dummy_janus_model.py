# models/dummy_janus_model.py
from models.base_model import BaseVisionModel
import logging
from typing import Tuple, Optional
import os

logger = logging.getLogger(__name__)

class JanusModel(BaseVisionModel):
    """Dummy Janus-Pro-1B model for when the actual model cannot be loaded."""
    
    def __init__(self, model_path: str = "deepseek-ai/Janus-Pro-1B"):
        """Initialize the dummy Janus-Pro-1B model."""
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
            "Error: Janus-Pro-1B model is not available due to import errors.\n\n"
            "Try installing the required dependencies:\n"
            "- pip install torch==2.6.0 torchvision==0.17.0\n"
            "- pip install git+https://github.com/huggingface/transformers.git\n"
            "- pip install accelerate\n"
            "- pip install bitsandbytes\n\n"
            "IMPORTANT: If you see a 'model type `multi_modality` not recognized' error,\n"
            "you MUST install the very latest transformers directly from GitHub:\n"
            "pip install git+https://github.com/huggingface/transformers.git\n\n"
            "Or continue using the Florence-2 model which is working correctly."
        )
        
        return message, None
    
    @classmethod
    def is_available(cls) -> bool:
        """This dummy model is always available as a fallback."""
        return True