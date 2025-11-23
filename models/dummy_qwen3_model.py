# models/dummy_qwen3_model.py
from models.base_model import BaseVisionModel
from typing import Tuple, Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class Qwen3Model(BaseVisionModel):
    """Dummy Qwen3 model for fallback when dependencies are missing."""
    
    def __init__(self, model_path: str = None):
        logger.warning("Initializing Dummy Qwen3Model")
        self.model_path = "dummy-qwen3"
        super().__init__()

    def _get_model_name(self) -> str:
        return "qwen3"

    def _setup_model(self) -> None:
        pass

    def analyze_image(self, image_path: str, quality: str = "standard", template_name: Optional[str] = None, 
                     template_variables: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str]]:
        return "Dummy Qwen3 Description: This is a placeholder description.", "This is a placeholder description."

    def analyze_images_batch(self, image_paths: List[str], quality: str = "standard",
                            template_name: Optional[str] = None, 
                            template_variables: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Optional[str]]]:
        return [self.analyze_image(path) for path in image_paths]
