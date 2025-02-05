from models.base_model import BaseVisionModel
import logging
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoProcessor

logger = logging.getLogger(__name__)

class JanusModel(BaseVisionModel):
    def analyze_image(self, image_path: str) -> Tuple[str, Optional[str]]:
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/janus-pro-1b",
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained("deepseek-ai/janus-pro-1b")
            
            outputs = self.model.generate(
                max_new_tokens=256,
                num_beams=5,
                length_penalty=1.0,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
            
            caption = self.processor.batch_decode(
                outputs,
                (function) clean_up_tokenization_spaces: Any,
                clean_up_tokenization_spaces=True
            )[0]
            
            caption = self.clean_output(caption)
            description = f"Description: {caption}"
            
            return description, caption
            
        except Exception as e:
            logger.error(f"Error analyzing image with Janus-Pro: {str(e)}")
            return "Error analyzing image.", None