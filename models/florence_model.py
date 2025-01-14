# models/florence_model.py
from .base_model import BaseVisionModel
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class Florence2Model(BaseVisionModel):
    def _setup_model(self) -> None:
        try:
            logger.info("Loading Florence-2 model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large",
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            ).to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-large",
                trust_remote_code=True
            )
            logger.info("Florence-2 model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load Florence-2 model: {str(e)}")
            raise

    def analyze_image(self, image_path: str) -> Tuple[str, Optional[str]]:
        try:
            image = Image.open(image_path)
            
            # Generate caption
            inputs = self.processor(
                text="<image>Describe this image in detail:",
                images=image,
                return_tensors="pt"
            ).to(self.device, self.torch_dtype)

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
                do_sample=False
            )
            
            caption = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            # Object detection
            inputs_od = self.processor(
                text="<OD>",
                images=image,
                return_tensors="pt"
            ).to(self.device, self.torch_dtype)

            generated_ids_od = self.model.generate(
                input_ids=inputs_od["input_ids"],
                pixel_values=inputs_od["pixel_values"],
                max_new_tokens=128,
                num_beams=3,
                do_sample=False
            )
            
            objects_text = self.processor.post_process_generation(
                self.processor.batch_decode(generated_ids_od, skip_special_tokens=False)[0],
                task="<OD>",
                image_size=(image.width, image.height)
            )
            
            caption = self.clean_output(caption)
            objects = [self.clean_output(obj) for obj in objects_text if obj.strip()]
            
            description = (
                f"Description: {caption}\n\n"
                f"Detected objects: {', '.join(objects)}"
            )
            
            return description, caption

        except Exception as e:
            logger.error(f"Error analyzing image with Florence-2: {str(e)}")
            return "Error analyzing image.", None