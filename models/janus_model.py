# models/janus_model.py
from .base_model import BaseVisionModel
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class JanusModel(BaseVisionModel):
    def _setup_model(self) -> None:
        try:
            logger.info("Loading Janus-Pro model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/janus-pro-1b",
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                use_safetensors=True
            ).to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(
                "deepseek-ai/janus-pro-1b",
                trust_remote_code=True
            )
            logger.info("Janus-Pro model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load Janus-Pro model: {str(e)}")
            raise

    def analyze_image(self, image_path: str) -> Tuple[str, Optional[str]]:
        try:
            image = Image.open(image_path)
            image = image.resize((384, 384))
            
            inputs = self.processor(
                images=image,
                text="Analyze this image in detail. Describe what you see and identify key objects.",
                return_tensors="pt",
                padding=True
            ).to(self.device, self.torch_dtype)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=5,
                length_penalty=1.0,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
            
            caption = self.processor.batch_decode(
                output_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            
            caption = self.clean_output(caption)
            description = f"Description: {caption}"
            
            return description, caption

        except Exception as e:
            logger.error(f"Error analyzing image with Janus-Pro: {str(e)}")
            return "Error analyzing image.", None