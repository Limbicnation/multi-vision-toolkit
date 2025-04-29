# models/dummy_florence_model.py
from models.base_model import BaseVisionModel
import logging
from typing import Tuple, Optional
import os

logger = logging.getLogger(__name__)

class Florence2Model(BaseVisionModel):
    """Dummy Florence2 model for when the actual model cannot be loaded."""
    
    def __init__(self):
        """Initialize the dummy Florence2 model."""
        logger.warning("Using dummy Florence2Model - actual model could not be loaded")
        # Skip dependency checks but call parent init
        super().__init__()
    
    def _setup_model(self) -> None:
        """Setup dummy model."""
        logger.warning("Using dummy Florence2Model with no actual model loaded")
        self.model = None
        self.processor = None
    
    def analyze_image(self, image_path: str, quality: str = "standard") -> Tuple[str, Optional[str]]:
        """Return a dummy analysis result."""
        if not os.path.exists(image_path):
            return "Error: Image file not found.", None
            
        message = (
            "Error: Florence-2 model is not available due to download/import errors.\n\n"
            "Try installing the required dependencies:\n"
            "- pip install torch==2.6.0 torchvision==0.17.0\n"
            "- pip install transformers>=4.30.0\n"
            "- pip install accelerate timm==0.9.2 einops\n"
            "- pip install safetensors\n\n"
            "The model will be downloaded automatically when properly configured."
        )
        
        return message, None
    
    @classmethod
    def is_available(cls) -> bool:
        """This dummy model is always available as a fallback."""
        return True