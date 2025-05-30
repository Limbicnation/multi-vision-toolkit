# models/qwen_model.py
from models.base_model import BaseVisionModel
import logging

logger = logging.getLogger(__name__) # Define logger early

import os
from typing import Tuple, Optional, Dict, List, Any
import importlib

try:
    import torch
except ImportError:
    logging.error("PyTorch is not installed. Please install PyTorch first.")
    torch = None # type: ignore

try:
    from PIL import Image
except ImportError:
    logger.warning("Pillow (PIL) not found. Image loading will fail.")
    Image = None # type: ignore

_QWEN_CLASS_AVAILABLE = False
Qwen2_5_VLForConditionalGeneration = None
AutoProcessor = None
AutoTokenizer = None # type: ignore
CLIPModel = None # type: ignore
CLIPProcessor = None # type: ignore
process_vision_info_fn = None # type: ignore

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
    _QWEN_CLASS_AVAILABLE = True
    logger.info("Successfully imported Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer.")
except ImportError as e:
    logger.error(
        f"Failed to import Qwen classes from transformers: {e}. "
        "Ensure transformers is installed from git source: 'pip install git+https://github.com/huggingface/transformers.git --upgrade'"
    )

try:
    from transformers import CLIPModel, CLIPProcessor
    logger.info("Successfully imported CLIPModel and CLIPProcessor for fallback.")
except ImportError as e:
    logger.warning(f"Failed to import CLIPModel or CLIPProcessor: {e}. Fallback to CLIP may not work.")

try:
    from qwen_vl_utils import process_vision_info
    process_vision_info_fn = process_vision_info
    logger.info("Successfully imported process_vision_info from qwen_vl_utils.")
except ImportError as e:
    logger.warning(
        f"Failed to import process_vision_info from qwen_vl_utils: {e}. "
        "Qwen model might not process inputs correctly. Install with: pip install qwen-vl-utils[decord]==0.0.8"
    )


class QwenModel(BaseVisionModel):
    """Qwen2.5-VL (non-AWQ) model implementation for image captioning."""
    
    REQUIRED_PACKAGES = {
        'transformers': 'transformers (latest from git)',
        'torch': 'torch',
        'PIL': 'Pillow',
        'qwen_vl_utils': 'qwen-vl-utils[decord]==0.0.8',
        'accelerate': 'accelerate',
        'flash_attn': 'flash-attn (optional, for performance, install with --no-build-isolation)'
    }

    def __init__(self, model_path: str = None):
        # Always default to the non-AWQ version due to autoawq complexities
        self.model_path = "Qwen/Qwen2.5-VL-3B-Instruct" 
        logger.info(f"Initializing QwenModel with non-AWQ model: {self.model_path}")
        if model_path is not None and model_path != self.model_path:
            logger.warning(f"Specified model_path '{model_path}' will be overridden by non-AWQ default '{self.model_path}'.")
            
        self._check_dependencies()
        self.tokenizer = None 
        super().__init__()


class QwenCaptioner(BaseVisionModel):
    """Qwen2.5-VL-7B-Captioner-Relaxed model implementation optimized for detailed captions."""
    
    REQUIRED_PACKAGES = {
        'transformers': 'transformers (latest from git)',
        'torch': 'torch',
        'PIL': 'Pillow',
        'qwen_vl_utils': 'qwen-vl-utils[decord]==0.0.8',
        'accelerate': 'accelerate',
        'bitsandbytes': 'bitsandbytes (for quantization)',
        'flash_attn': 'flash-attn (optional, for performance, install with --no-build-isolation)'
    }

    def __init__(self, model_path: str = None, use_quantization: str = None):
        self.model_path = model_path or "Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed"
        self.use_quantization = use_quantization or "8bit"  # Default to 8-bit for 23.5GB VRAM
        logger.info(f"Initializing QwenCaptioner with model: {self.model_path}, quantization: {self.use_quantization}")
        
        self._check_dependencies()
        self.tokenizer = None
        super().__init__()

    def load(self):
        """Load the model and return self for method chaining."""
        if not hasattr(self, 'model') or self.model is None:
            self._setup_model()
        return self

    def _setup_model(self) -> None:
        """Override to use captioner-specific setup method."""
        self._setup_model_captioner()

    def caption_image(self, image_path: str, quality: str = "detailed") -> str:
        """Caption a single image with the captioner model."""
        description, caption = self.analyze_image(image_path, quality)
        return caption if caption else description

    def caption_batch(self, image_paths: List[str], quality: str = "detailed") -> List[str]:
        """Caption multiple images in batch."""
        results = self.analyze_images_batch(image_paths, quality)
        return [caption if caption else description for description, caption in results]

    def get_instruction_for_quality_captioner(self, quality: str) -> str:
        """Get captioner-specific instructions optimized for the 7B Captioner model."""
        if quality == "standard":
            return "Describe this image concisely."
        elif quality == "detailed":
            return "Generate a comprehensive, detailed caption for this image. Include objects, people, setting, colors, mood, and composition details."
        elif quality == "creative":
            return "Create an artistic, evocative description of this image that captures both literal elements and emotional atmosphere."
        return "Caption this image."

    def get_generation_params_captioner(self, quality: str) -> dict:
        """Get captioner-specific generation parameters optimized for the 7B model."""
        if quality == "standard":
            return {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "repetition_penalty": 1.1
            }
        elif quality == "detailed":
            return {
                "max_new_tokens": 300,
                "temperature": 0.6,
                "top_p": 0.85,
                "repetition_penalty": 1.15,
                "do_sample": True,
                "num_beams": 3
            }
        elif quality == "creative":
            return {
                "max_new_tokens": 200,
                "temperature": 0.8,
                "top_p": 0.95,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "top_k": 50
            }
        return {"max_new_tokens": 100, "temperature": 0.7, "top_p": 0.9, "do_sample": True}

    def analyze_image(self, image_path: str, quality: str = "detailed") -> Tuple[str, Optional[str]]:
        """Override analyze_image to use captioner-specific methods."""
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return "Error: Image file not found.", None
        
        try:
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return "Error: Failed to load or process image.", None

        if getattr(self, '_using_fallback', False) or not all([self.model, self.processor, self.tokenizer, _QWEN_CLASS_AVAILABLE, process_vision_info_fn]):
            logger.info("Using fallback CLIP model for image analysis (Qwen components not fully available or in fallback mode).")
            return self._analyze_with_clip(pil_image, quality)

        # Use captioner-specific instruction
        instruction = self.get_instruction_for_quality_captioner(quality)
        
        messages = [
            {"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": instruction}]}
        ]

        try:
            text_for_template = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs_processed, video_inputs_processed = process_vision_info_fn(messages)
            
            inputs = self.processor(
                text=[text_for_template],
                images=image_inputs_processed,
                videos=video_inputs_processed,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # Use captioner-specific generation parameters
            generation_params = self.get_generation_params_captioner(quality)
            
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **generation_params)
            
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
            caption = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )[0]
            caption = self.clean_output(caption)
            
            model_name = "Qwen2.5-VL-7B-Captioner-Relaxed"
            description = f"Description: {caption}\n\nGenerated by: {model_name}"
            return description, caption

        except Exception as e:
            logger.error(f"Error generating caption with QwenCaptioner: {str(e)}")
            return self._analyze_with_clip(pil_image, quality)

    @classmethod
    def _check_dependencies(cls) -> None:
        missing_packages = []
        for package, pip_name in cls.REQUIRED_PACKAGES.items():
            if package == 'flash_attn': 
                try:
                    importlib.import_module(package)
                except ImportError:
                    logger.warning("flash_attn not found. For optimal performance, install with 'pip install flash-attn --no-build-isolation'.")
                continue

            try:
                importlib.import_module(package)
                if package == 'transformers' and not _QWEN_CLASS_AVAILABLE:
                    missing_packages.append((package, f"{pip_name} (Qwen2_5_VLForConditionalGeneration class not found. Ensure latest git version.)"))
                elif package == 'qwen_vl_utils' and process_vision_info_fn is None:
                     missing_packages.append((package, pip_name))
            except ImportError:
                missing_packages.append((package, pip_name))
        
        if missing_packages:
            install_commands = []
            for pkg, name in missing_packages:
                if "transformers" in name:
                    install_commands.append("pip install git+https://github.com/huggingface/transformers.git --upgrade")
                else:
                    install_commands.append(f"pip install {name}")
            
            error_msg = (
                f"Missing required packages/classes for QwenModel: {', '.join(pkg[0] for pkg in missing_packages)}\n"
                f"Please install or update them.\nExample install commands:\n" + "\n".join(install_commands)
            )
            logger.error(error_msg)

    def _load_clip_as_fallback(self, reason: str) -> None:
        logger.warning(f"Attempting to load CLIP model as a fallback due to: {reason}")
        try:
            global CLIPModel, CLIPProcessor
            if CLIPModel is None or CLIPProcessor is None:
                from transformers import CLIPModel as DynamicCLIPModel, CLIPProcessor as DynamicCLIPProcessor
                CLIPModel = DynamicCLIPModel
                CLIPProcessor = DynamicCLIPProcessor
                if CLIPModel is None or CLIPProcessor is None:
                    raise ImportError("CLIPModel/CLIPProcessor not available for fallback.")

            fallback_model_id = "openai/clip-vit-base-patch32"
            clip_dtype_to_use = self.torch_dtype if self.device.startswith('cuda') else torch.float32
            logger.info(f"Loading fallback CLIP model: {fallback_model_id} with dtype: {clip_dtype_to_use} on device: {self.device}")

            self.model = CLIPModel.from_pretrained(fallback_model_id, torch_dtype=clip_dtype_to_use).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(fallback_model_id)
            self.tokenizer = None 
            
            logger.info(f"Successfully loaded CLIP model as fallback: {fallback_model_id} to device {self.device} with dtype {self.model.dtype}")
            self._using_fallback = True
        except Exception as fallback_error:
            logger.error(f"Failed to load CLIP fallback model: {str(fallback_error)}")
            raise RuntimeError(f"Qwen model setup failed, and fallback CLIP model also failed to load: {fallback_error}") from fallback_error

    def _setup_model(self) -> None:
        self._using_fallback = False
        try:
            if not _QWEN_CLASS_AVAILABLE or Qwen2_5_VLForConditionalGeneration is None:
                logger.error("Qwen2_5_VLForConditionalGeneration class not available. Falling back.")
                self._load_clip_as_fallback(reason="Qwen2_5_VLForConditionalGeneration class not found.")
                return

            if process_vision_info_fn is None:
                logger.error("process_vision_info from qwen_vl_utils not available. Falling back.")
                self._load_clip_as_fallback(reason="process_vision_info_fn not available.")
                return
            
            logger.info(f"Loading Qwen2.5-VL (non-AWQ) model: {self.model_path}")
            
            model_kwargs: Dict[str, Any] = {
                "torch_dtype": self.torch_dtype,
                "trust_remote_code": True
            }
            
            if self.device.startswith('cuda'):
                model_kwargs["device_map"] = torch.device(self.device)
                logger.info(f"Setting device_map to torch.device('{self.device}') for CUDA.")
            elif self.device in ["cpu", "mps"]:
                model_kwargs["device_map"] = self.device
                logger.info(f"Setting device_map to '{self.device}'.")
            else:
                model_kwargs["device_map"] = "auto" # Fallback for other scenarios
                logger.info("Setting device_map to 'auto'.")

            if self.device.startswith('cuda') and (self.torch_dtype == torch.bfloat16 or self.torch_dtype == torch.float16):
                try:
                    import flash_attn 
                    logger.info("Attempting to enable Flash Attention 2 for Qwen model.")
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                except ImportError:
                    logger.warning("flash_attn library not found. Flash Attention 2 cannot be enabled for Qwen. Install with 'pip install flash-attn --no-build-isolation'")
                except Exception as e: # Catch other potential errors if attn_implementation is not supported by the transformers version
                    logger.warning(f"Could not enable Flash Attention 2 for Qwen: {e}. Proceeding without it.")
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path, **model_kwargs)
            logger.info(f"Successfully loaded Qwen model: {self.model_path} with kwargs: {model_kwargs}")
            
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            logger.info(f"Successfully loaded Qwen processor and tokenizer for {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to initialize Qwen model: {str(e)}")
            self._load_clip_as_fallback(reason=f"Qwen (non-AWQ) load failed: {e}")

    def _setup_model_captioner(self) -> None:
        """Setup method specifically for QwenCaptioner with quantization support."""
        self._using_fallback = False
        try:
            if not _QWEN_CLASS_AVAILABLE or Qwen2_5_VLForConditionalGeneration is None:
                logger.error("Qwen2_5_VLForConditionalGeneration class not available. Falling back.")
                self._load_clip_as_fallback(reason="Qwen2_5_VLForConditionalGeneration class not found.")
                return

            if process_vision_info_fn is None:
                logger.error("process_vision_info from qwen_vl_utils not available. Falling back.")
                self._load_clip_as_fallback(reason="process_vision_info_fn not available.")
                return
            
            logger.info(f"Loading Qwen2.5-VL-7B-Captioner-Relaxed model: {self.model_path}")
            
            model_kwargs: Dict[str, Any] = {
                "torch_dtype": self.torch_dtype,
                "trust_remote_code": True
            }
            
            # Apply quantization if specified
            if self.use_quantization == "4bit":
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                model_kwargs["quantization_config"] = bnb_config
                logger.info("Using 4-bit quantization")
            elif self.use_quantization == "8bit":
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["quantization_config"] = bnb_config
                logger.info("Using 8-bit quantization")
            
            if self.device.startswith('cuda'):
                model_kwargs["device_map"] = "auto"
                logger.info("Setting device_map to 'auto' for CUDA with quantization support.")
            elif self.device in ["cpu", "mps"]:
                model_kwargs["device_map"] = self.device
                logger.info(f"Setting device_map to '{self.device}'.")
            else:
                model_kwargs["device_map"] = "auto"
                logger.info("Setting device_map to 'auto'.")

            # Flash Attention for performance
            if self.device.startswith('cuda') and (self.torch_dtype == torch.bfloat16 or self.torch_dtype == torch.float16):
                try:
                    import flash_attn 
                    logger.info("Attempting to enable Flash Attention 2 for QwenCaptioner.")
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                except ImportError:
                    logger.warning("flash_attn library not found. Flash Attention 2 cannot be enabled.")
                except Exception as e:
                    logger.warning(f"Could not enable Flash Attention 2: {e}. Proceeding without it.")
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path, **model_kwargs)
            logger.info(f"Successfully loaded QwenCaptioner model: {self.model_path} with kwargs: {model_kwargs}")
            
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            logger.info(f"Successfully loaded QwenCaptioner processor and tokenizer for {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to initialize QwenCaptioner model: {str(e)}")
            self._load_clip_as_fallback(reason=f"QwenCaptioner load failed: {e}")

    def get_instruction_for_quality(self, quality: str) -> str:
        """Get appropriate instruction based on quality setting"""
        if quality == "standard":
            return "Describe this image briefly and concisely."
        elif quality == "detailed":
            return "Provide a detailed description of this image, including objects, people, colors, background, and any notable features. Be comprehensive."
        elif quality == "creative":
            return "Describe this image in a creative, imaginative way. Use evocative language and vivid descriptions. Feel free to interpret what you see."
        return "Describe this image."  # Default fallback
    
    def get_generation_params(self, quality: str) -> dict:
        """Get generation parameters based on quality mode"""
        if quality == "standard":
            return {
                "max_new_tokens": 75,      # Keep standard output concise
                "temperature": 0.7,        # Balanced randomness
                "top_p": 0.9,              # Standard filtering
                "do_sample": True
            }
        elif quality == "detailed":
            return {
                "max_new_tokens": 250,     # Much longer output
                "temperature": 0.6,        # Slightly lower temperature for factuality
                "top_p": 0.85,             # Slightly stricter filtering
                "repetition_penalty": 1.2, # Discourage repetition
                "do_sample": True,
                "num_beams": 3             # Add beam search for better quality
            }
        elif quality == "creative":
            return {
                "max_new_tokens": 175,     # Medium length for creativity
                "temperature": 0.9,        # Higher temperature for more creativity
                "top_p": 0.95,             # Higher top_p for more variety
                "repetition_penalty": 1.0, # Normal repetition handling
                "do_sample": True,
                "top_k": 40                # Add top_k sampling for creative diversity
            }
        # Default to standard if unknown quality provided
        return {"max_new_tokens": 75, "temperature": 0.7, "top_p": 0.9, "do_sample": True}

    def analyze_image(self, image_path: str, quality: str = "standard") -> Tuple[str, Optional[str]]:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return "Error: Image file not found.", None
        
        try:
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return "Error: Failed to load or process image.", None

        if getattr(self, '_using_fallback', False) or not all([self.model, self.processor, self.tokenizer, _QWEN_CLASS_AVAILABLE, process_vision_info_fn]):
            logger.info("Using fallback CLIP model for image analysis (Qwen components not fully available or in fallback mode).")
            return self._analyze_with_clip(pil_image, quality)

        # Get quality-specific instruction
        instruction = self.get_instruction_for_quality(quality)
        
        messages = [
            {"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": instruction}]}
        ]

        try:
            text_for_template = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs_processed, video_inputs_processed = process_vision_info_fn(messages)
            
            inputs = self.processor(
                text=[text_for_template],
                images=image_inputs_processed,
                videos=video_inputs_processed,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # Use quality-specific generation parameters
            generation_params = self.get_generation_params(quality)
            
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **generation_params)
            
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
            caption = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True  # Try setting to True to help with encoding issues
            )[0]
            caption = self.clean_output(caption)
            
            model_name = "Qwen2.5-VL (non-AWQ)"
            description = f"Description: {caption}\n\nGenerated by: {model_name}"
            return description, caption

        except Exception as e:
            logger.error(f"Error generating caption with Qwen: {str(e)}")
            return self._analyze_with_clip(pil_image, quality)

    def analyze_images_batch(self, image_paths: List[str], quality: str = "standard") -> List[Tuple[str, Optional[str]]]:
        if not image_paths:
            return []

        results: List[Optional[Tuple[str, Optional[str]]]] = [None] * len(image_paths)
        pil_images_to_process: List[Tuple[int, Image.Image]] = []  # Stores (original_index, pil_image)

        for i, image_path in enumerate(image_paths):
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                results[i] = (f"Error: Image file not found at {image_path}.", None)
                continue
            try:
                pil_image = Image.open(image_path)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                pil_images_to_process.append((i, pil_image))
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {str(e)}")
                results[i] = (f"Error: Failed to load image {image_path}: {str(e)}.", None)
        
        actual_pil_images = [img for _, img in pil_images_to_process]
        original_indices_for_processing = [idx for idx, _ in pil_images_to_process]

        if not actual_pil_images:
            return [res if res is not None else ("Error: No valid images to process.", None) for res in results]

        if getattr(self, '_using_fallback', False) or not all([self.model, self.processor, self.tokenizer, _QWEN_CLASS_AVAILABLE, process_vision_info_fn]):
            logger.info("Using fallback CLIP model for batch image analysis.")
            clip_batch_results = self._analyze_batch_with_clip(actual_pil_images, quality)
            for i, res_tuple in enumerate(clip_batch_results):
                original_idx = original_indices_for_processing[i]
                results[original_idx] = res_tuple
            return [res if res is not None else ("Error: Fallback processing issue.", None) for res in results]

        # --- Qwen Batch Processing ---
        texts_for_template_batch: List[str] = []
        processed_image_inputs_batch: List[Any] = [] 
        
        # Keep track of original indices that successfully make it through Qwen pre-processing
        valid_original_indices_for_qwen_output: List[int] = []

        for i, pil_image in enumerate(actual_pil_images):
            current_original_idx = original_indices_for_processing[i]
            instruction = self.get_instruction_for_quality(quality)
            current_messages = [
                {"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": instruction}]}
            ]
            
            try:
                text_for_template = self.tokenizer.apply_chat_template(current_messages, tokenize=False, add_generation_prompt=True)
                img_inputs_for_current_msg, _ = process_vision_info_fn(current_messages)
                
                if img_inputs_for_current_msg and len(img_inputs_for_current_msg) == 1:
                    texts_for_template_batch.append(text_for_template)
                    processed_image_inputs_batch.extend(img_inputs_for_current_msg) 
                    valid_original_indices_for_qwen_output.append(current_original_idx)
                else:
                    err_msg = f"Error: Qwen pre-processing failed for {image_paths[current_original_idx]} (unexpected output from process_vision_info_fn)."
                    logger.warning(err_msg)
                    results[current_original_idx] = (err_msg, None)
            except Exception as e:
                err_msg = f"Error: Qwen pre-processing failed for {image_paths[current_original_idx]} ({str(e)})."
                logger.error(err_msg)
                results[current_original_idx] = (err_msg, None)

        if not texts_for_template_batch:
            logger.info("No images were successfully pre-processed for Qwen batch.")
            return [res if res is not None else ("Error: Qwen pre-processing failed for all images.", None) for res in results]

        try:
            inputs = self.processor(
                text=texts_for_template_batch,
                images=processed_image_inputs_batch,
                videos=None, 
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # Use quality-specific generation parameters
            generation_params = self.get_generation_params(quality)
            
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **generation_params)
            
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            captions_batch_list = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True  # Try setting to True to help with encoding issues
            )
            
            model_name_str = "Qwen2.5-VL (non-AWQ)"
            for i, caption_str in enumerate(captions_batch_list):
                clean_caption = self.clean_output(caption_str)
                description = f"Description: {clean_caption}\n\nGenerated by: {model_name_str}"
                current_original_idx = valid_original_indices_for_qwen_output[i]
                results[current_original_idx] = (description, clean_caption)

        except Exception as e:
            err_msg_batch = f"Error: Qwen batch generation failed ({str(e)})."
            logger.error(err_msg_batch)
            for original_idx in valid_original_indices_for_qwen_output:
                if results[original_idx] is None: # Only update if not already set by individual pre-processing error
                     results[original_idx] = (err_msg_batch, None)
        
        return [res if res is not None else ("Error: Unknown processing issue.", None) for res in results]

    def get_clip_description_for_quality(self, top_category: str, scores: list, quality: str = "standard") -> str:
        """Get appropriately detailed CLIP description based on quality setting"""
        confidence = scores[0][1] * 100
        
        if quality == "detailed":
            other_elements = [f"{s_cat} ({s_prob*100:.1f}%)" for i, (s_cat, s_prob) in enumerate(scores[1:3])]
            caption_str = f"This image appears to be a {top_category} (confidence: {confidence:.1f}%). "
            if other_elements: 
                caption_str += f"It may also contain elements of {' and '.join(other_elements)}."
                
            # Add more detailed description based on category
            if top_category == "landscape":
                caption_str += " The landscape features natural elements with distinct composition and lighting."
            elif top_category == "portrait":
                caption_str += " The portrait shows a subject with distinct facial features against a complementary background."
            elif top_category == "food":
                caption_str += " The food is artfully presented with appealing colors and textures."
            elif top_category == "animal":
                caption_str += " The animal is captured in its environment with characteristic features visible."
            elif top_category == "building":
                caption_str += " The architectural structure displays distinctive design elements and proportions."
            elif top_category == "people":
                caption_str += " The individuals are engaged in activities that reveal their relationships and setting."
            elif top_category == "abstract":
                caption_str += " The abstract elements feature visual patterns, textures and color arrangements that create aesthetic interest."
            elif top_category == "illustration":
                caption_str += " The illustration demonstrates artistic technique with intentional stylistic choices."
                
        elif quality == "creative":
            caption_str = f"A captivating {top_category} scene that draws the viewer in. "
            if top_category == "landscape":
                caption_str += "The natural beauty unfolds with layers of color and texture, inviting exploration beyond the visible boundaries."
            elif top_category == "portrait":
                caption_str += "The subject's presence tells a story through expression, with eyes that seem to hold hidden narratives and emotions."
            elif top_category == "food":
                caption_str += "A feast for the eyes as much as for the palate, with colors and textures that awaken the imagination."
            elif top_category == "animal":
                caption_str += "The creature's character is a compelling focal point, revealing the wild spirit that dwells within all living beings."
            elif top_category == "building":
                caption_str += "The structure stands as a testament to human creativity, its form both challenging and harmonizing with the surroundings."
            elif top_category == "people":
                caption_str += "Human moments frozen in time, each figure part of an unfolding story that invites countless interpretations."
            elif top_category == "abstract":
                caption_str += "Forms and colors dance together in a composition that speaks to emotions rather than literal representation."
            elif top_category == "illustration":
                caption_str += "The artist's vision manifests through deliberate strokes and stylistic choices that transport the viewer to imagined worlds."
        else:  # standard
            caption_str = f"This image shows a {top_category}."
            
        return caption_str

    def _analyze_with_clip(self, image: Image.Image, quality: str = "standard") -> Tuple[str, Optional[str]]:
        logger.info("Using CLIP fallback for image analysis (invoked from _analyze_with_clip)")
        try:
            global CLIPModel, CLIPProcessor
            if CLIPModel is None or CLIPProcessor is None:
                from transformers import CLIPProcessor as DynamicCLIPProcessor, CLIPModel as DynamicCLIPModel
                CLIPModel = DynamicCLIPModel 
                CLIPProcessor = DynamicCLIPProcessor
                if CLIPModel is None or CLIPProcessor is None:
                     logger.error("Failed to import CLIP models for fallback.")
                     return "Error: CLIP Fallback components not available.", None
            
            if not (hasattr(self, 'model') and isinstance(self.model, CLIPModel) and \
                    hasattr(self, 'processor') and isinstance(self.processor, CLIPProcessor) and \
                    getattr(self, '_using_fallback', False)):
                
                fallback_model_id = "openai/clip-vit-base-patch32"
                clip_dtype_to_use = self.torch_dtype if self.device.startswith('cuda') else torch.float32
                logger.info(f"Loading/Re-loading CLIP model for fallback: {fallback_model_id} with dtype: {clip_dtype_to_use} on device: {self.device}")
                
                self.model = CLIPModel.from_pretrained(fallback_model_id, torch_dtype=clip_dtype_to_use).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(fallback_model_id)
                self._using_fallback = True
            else:
                logger.info("Reusing existing CLIP model and processor for fallback analysis.")

            inputs = self.processor(
                text=["a photo of a landscape", "a portrait", "a photo of food", 
                      "a photo of an animal", "a photo of a building", "a photo of people", "an abstract image", "a drawing or illustration"],
                images=image, # Single image for this method
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model(**inputs)
                
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).tolist()[0]
            
            categories = ["landscape", "portrait", "food", "animal", "building", "people", "abstract", "illustration"]
            scores = sorted(list(zip(categories, probs)), key=lambda x: x[1], reverse=True)
            
            top_category = scores[0][0]
            
            # Use quality-specific CLIP description generator
            caption_str = self.get_clip_description_for_quality(top_category, scores, quality)
                
            description = f"Description: {caption_str}\n\nGenerated by: CLIP (Fallback Mode)"
            return description, caption_str
            
        except Exception as e:
            logger.error(f"Error in CLIP fallback analysis: {str(e)}")
            return f"Image analysis failed with CLIP fallback. Error: {str(e)}", None

    def _analyze_batch_with_clip(self, pil_images: List[Image.Image], quality: str = "standard") -> List[Tuple[str, Optional[str]]]:
        if not pil_images:
            return []
        
        logger.info(f"Using CLIP fallback for batch analysis of {len(pil_images)} images.")
        batch_results: List[Tuple[str, Optional[str]]] = []

        try:
            global CLIPModel, CLIPProcessor # Ensure they are accessible
            if CLIPModel is None or CLIPProcessor is None:
                from transformers import CLIPProcessor as DynamicCLIPProcessor, CLIPModel as DynamicCLIPModel
                CLIPModel = DynamicCLIPModel 
                CLIPProcessor = DynamicCLIPProcessor
                if CLIPModel is None or CLIPProcessor is None:
                    logger.error("Failed to import CLIP models for batch fallback.")
                    return [("Error: CLIP Fallback components not available.", None)] * len(pil_images)
            
            if not (hasattr(self, 'model') and isinstance(self.model, CLIPModel) and \
                    hasattr(self, 'processor') and isinstance(self.processor, CLIPProcessor) and \
                    getattr(self, '_using_fallback', False)):
                fallback_model_id = "openai/clip-vit-base-patch32"
                clip_dtype_to_use = self.torch_dtype if self.device.startswith('cuda') else torch.float32
                self.model = CLIPModel.from_pretrained(fallback_model_id, torch_dtype=clip_dtype_to_use).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(fallback_model_id)
                self._using_fallback = True
            
            text_prompts = ["a photo of a landscape", "a portrait", "a photo of food", 
                            "a photo of an animal", "a photo of a building", "a photo of people", 
                            "an abstract image", "a drawing or illustration"]
            
            inputs = self.processor(
                text=text_prompts,
                images=pil_images, # Pass list of PIL images
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model(**inputs)
                
            logits_per_image = outputs.logits_per_image # Shape: (batch_size, num_text_prompts)
            probs_batch = logits_per_image.softmax(dim=1).tolist() # List of lists of probabilities
            
            categories = ["landscape", "portrait", "food", "animal", "building", "people", "abstract", "illustration"]
            
            for probs_single_image in probs_batch:
                scores = sorted(list(zip(categories, probs_single_image)), key=lambda x: x[1], reverse=True)
                top_category = scores[0][0]
                
                # Use quality-specific CLIP description generator
                caption_str = self.get_clip_description_for_quality(top_category, scores, quality)
                
                description = f"Description: {caption_str}\n\nGenerated by: CLIP (Fallback Mode)"
                batch_results.append((description, caption_str))
                
            return batch_results
                
        except Exception as e:
            logger.error(f"Error in CLIP batch fallback analysis: {str(e)}")
            return [(f"Image analysis failed with CLIP fallback. Error: {str(e)}", None)] * len(pil_images)

    def _preprocess_image(self, image: Image.Image) -> Any:
        logger.debug("QwenModel._preprocess_image called, but typically handled by processor/tokenizer.")
        return image 

    @classmethod
    def is_available(cls) -> bool:
        if not _QWEN_CLASS_AVAILABLE:
            logger.warning("Qwen2_5_VLForConditionalGeneration class not found. Qwen model not available.")
            return False
        if process_vision_info_fn is None: # Check the imported function reference
            logger.warning("process_vision_info from qwen_vl_utils not found. Qwen model may not be fully available.")
            # Consider returning False if qwen_vl_utils is absolutely critical for any operation
        return True
