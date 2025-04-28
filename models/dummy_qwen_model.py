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
        """Return a dummy analysis result."""
        if not os.path.exists(image_path):
            return "Error: Image file not found.", None
            
        message = (
            "Error: Qwen model is not available due to import errors.\n\n"
            "Try installing the required dependencies:\n"
            "- pip install torch==2.6.0 torchvision==0.17.0\n"
            "- pip install transformers==4.36.2\n"
            "- pip install qwen-vl-utils[decord]==0.0.8\n"
            "- pip install accelerate einops timm==0.9.2\n"
            "- pip install ftfy regex tqdm safetensors\n\n"
            "Or try using another model like 'janus'"
        )
        
        return message, None
    
    @classmethod
    def is_available(cls) -> bool:
        """This dummy model is always available as a fallback."""
        return True