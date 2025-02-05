from models.base_model import BaseVisionModel
import logging
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from typing import Tuple, Optional

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the token
hf_token = os.getenv('HF_TOKEN')

logger = logging.getLogger(__name__)

class JanusModel(BaseVisionModel):
    def __init__(self):
        super().__init__()

    def _setup_model(self) -> None:
        try:
            logger.info("Loading Janus-Pro model...")
            model_path = "janhq/janus-1-7b"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                device_map="balanced",
                trust_remote_code=True,
                load_in_8bit=True
            )
            
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Janus model: {str(e)}")
            raise

    def analyze_image(self, image_path: str) -> Tuple[str, Optional[str]]:
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(
                images=image,
                text="Describe this image in detail.",
                return_tensors="pt"
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=5,
                    length_penalty=1.0,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    use_cache=True
                )
            
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            caption = self.clean_output(caption)
            description = f"Description: {caption}"
            
            return description, caption
            
        except Exception as e:
            logger.error(f"Error analyzing image with Janus-Pro: {str(e)}")
            return "Error analyzing image.", None