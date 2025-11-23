# models/qwen3_model.py
from models.base_model import BaseVisionModel
import logging
import os
from typing import Tuple, Optional, Dict, List, Any
import importlib
import torch

logger = logging.getLogger(__name__)

# Set PyTorch memory allocation config to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class Qwen3Model(BaseVisionModel):
    """Qwen3-VL-4B-Instruct model implementation."""
    
    REQUIRED_PACKAGES = {
        'transformers': 'transformers',
        'torch': 'torch',
        'PIL': 'Pillow',
        'accelerate': 'accelerate'
    }

    def __init__(self, model_path: str = None):
        # Use local model by default if available, else remote
        default_local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "weights", "Qwen3-VL-4B-Instruct")
        
        if model_path is None:
            if os.path.exists(default_local_path):
                self.model_path = default_local_path
                logger.info(f"Using local Qwen3 model at: {self.model_path}")
            else:
                self.model_path = "Qwen/Qwen3-VL-4B-Instruct"
                logger.info(f"Local model not found, using remote: {self.model_path}")
        else:
            self.model_path = model_path
            logger.info(f"Using specified model path: {self.model_path}")
            
        self._check_dependencies()
        self.tokenizer = None 
        super().__init__()

    def _get_model_name(self) -> str:
        """Get the model name for template system integration."""
        try:
            from templates.template_manager import ModelNames
            return getattr(ModelNames, "QWEN3", "qwen3")
        except ImportError:
            return "qwen3"

    def _setup_model(self) -> None:
        """Setup the Qwen3 model."""
        try:
            # Try to import specific class as recommended in docs
            try:
                from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer
                ModelClass = Qwen3VLForConditionalGeneration
                logger.info("Successfully imported Qwen3VLForConditionalGeneration")
            except ImportError:
                logger.warning("Qwen3VLForConditionalGeneration not found, falling back to AutoModelForVision2Seq")
                from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
                ModelClass = AutoModelForVision2Seq

            logger.info(f"Loading Qwen3 model: {self.model_path}")
            
            model_kwargs: Dict[str, Any] = {
                "torch_dtype": self.torch_dtype,
                "trust_remote_code": True,
            }
            
            if self.device.startswith('cuda'):
                model_kwargs["device_map"] = "auto"
            elif self.device in ["cpu", "mps"]:
                model_kwargs["device_map"] = self.device
            else:
                model_kwargs["device_map"] = "auto"

            # Use eager attention for stability unless flash attention is specifically requested/configured
            # The docs recommend flash_attention_2 but we stick to eager for compatibility unless user overrides
            model_kwargs["attn_implementation"] = "eager"
            
            self.model = ModelClass.from_pretrained(self.model_path, **model_kwargs)
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            
            logger.info(f"Successfully loaded Qwen3 model: {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to initialize Qwen3 model: {str(e)}")
            raise e

    def analyze_image(self, image_path: str, quality: str = "standard", template_name: Optional[str] = None, 
                     template_variables: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str]]:
        """Analyze a single image."""
        if not os.path.exists(image_path):
            return "Error: Image file not found.", None
        
        try:
            from PIL import Image
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                
            # Determine prompt
            if template_name is not None:
                prompt = self.get_prompt_from_template(quality, template_name, template_variables)
            else:
                prompt = self.get_prompt_from_template(quality, f"caption_{quality}", template_variables)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Use the method from the docs: apply_chat_template with return_tensors="pt"
            # This handles image processing and tokenization in one go if supported by the processor
            try:
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
            except Exception as e:
                # Fallback to manual processing if apply_chat_template doesn't support images directly in this version
                logger.debug(f"apply_chat_template failed, falling back to manual processing: {e}")
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(
                    text=[text],
                    images=pil_image,
                    padding=True,
                    return_tensors="pt",
                )

            inputs = inputs.to(self.model.device)
            
            # Generation args based on docs and quality
            # Docs recommend: top_p=0.8, top_k=20, temperature=0.7, repetition_penalty=1.0, presence_penalty=1.5
            gen_kwargs = {
                "max_new_tokens": 512 if quality == "detailed" else 128,
                "repetition_penalty": 1.0,
            }
            
            if quality == "creative":
                gen_kwargs.update({
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                })
            elif quality == "detailed":
                gen_kwargs.update({
                    "do_sample": True, # Docs imply sampling is used
                    "temperature": 0.4,
                    "top_p": 0.8,
                })
            else: # standard
                gen_kwargs.update({
                    "do_sample": False, # Greedy for standard/concise
                })

            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            clean_text = self.clean_output(output_text)
            
            return f"Description: {clean_text}\n\nGenerated by: Qwen3-VL-4B-Instruct", clean_text

        except Exception as e:
            logger.error(f"Error analyzing image with Qwen3: {e}")
            return f"Error: {str(e)}", None

    def analyze_images_batch(self, image_paths: List[str], quality: str = "standard",
                            template_name: Optional[str] = None, 
                            template_variables: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Optional[str]]]:
        """Analyze multiple images in batch."""
        # Simple sequential implementation for now
        results = []
        for path in image_paths:
            results.append(self.analyze_image(path, quality, template_name, template_variables))
        return results

    @classmethod
    def _check_dependencies(cls) -> None:
        missing = []
        for pkg, name in cls.REQUIRED_PACKAGES.items():
            try:
                importlib.import_module(pkg)
            except ImportError:
                missing.append(name)
        
        if missing:
            logger.warning(f"Missing packages for Qwen3: {', '.join(missing)}")
