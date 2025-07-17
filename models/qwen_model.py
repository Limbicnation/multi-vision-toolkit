# models/qwen_model.py
from models.base_model import BaseVisionModel
import logging

logger = logging.getLogger(__name__) # Define logger early

import os
from typing import Tuple, Optional, Dict, List, Any
import importlib

# Set PyTorch memory allocation config to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Disable flash attention globally to prevent symbol conflicts
os.environ["DISABLE_FLASH_ATTENTION"] = "1"
os.environ["FLASH_ATTENTION_SKIP_CUDA_CHECK"] = "1"

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

def safe_import_transformers():
    """Safely import transformers with flash attention guards."""
    global _QWEN_CLASS_AVAILABLE, Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
    
    try:
        # Check for flash attention conflicts before importing
        import importlib.util
        if importlib.util.find_spec("flash_attn"):
            logger.warning("Flash attention detected - may cause symbol conflicts, proceeding with caution")
        
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
        _QWEN_CLASS_AVAILABLE = True
        logger.info("Successfully imported Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer.")
        return True
        
    except ImportError as e:
        error_msg = str(e)
        if "flash_attn" in error_msg or "undefined symbol" in error_msg:
            logger.error(f"Flash attention symbol conflict detected: {error_msg}")
            logger.info("Attempting import without flash attention...")
            # Try again with more aggressive flash attention disabling
            os.environ["USE_FLASH_ATTENTION"] = "0"
            os.environ["FLASH_ATTN_DISABLE"] = "1"
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
                _QWEN_CLASS_AVAILABLE = True
                logger.info("Successfully imported transformers without flash attention.")
                return True
            except Exception as e2:
                logger.error(f"Still failed after disabling flash attention: {e2}")
                _QWEN_CLASS_AVAILABLE = False
                return False
        else:
            logger.error(f"Failed to import Qwen classes from transformers: {e}")
            logger.error("This may indicate the transformers version doesn't support Qwen2_5_VLForConditionalGeneration")
            logger.error("Please update transformers: pip install git+https://github.com/huggingface/transformers.git")
            _QWEN_CLASS_AVAILABLE = False
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error importing transformers: {e}")
        _QWEN_CLASS_AVAILABLE = False
        return False

# Attempt safe import
safe_import_transformers()

def safe_import_clip():
    """Safely import CLIP models with flash attention guards."""
    global CLIPModel, CLIPProcessor
    
    try:
        from transformers import CLIPModel, CLIPProcessor
        logger.info("Successfully imported CLIPModel and CLIPProcessor for fallback.")
        return True
        
    except ImportError as e:
        logger.warning(f"Failed to import CLIPModel or CLIPProcessor: {e}. Fallback to CLIP may not work.")
        return False
        
    except Exception as e:
        error_msg = str(e)
        if "flash_attn" in error_msg or "undefined symbol" in error_msg:
            logger.warning(f"CLIP import flash attention conflict: {e}. Attempting workaround...")
            try:
                # Try importing individual components
                import transformers
                CLIPModel = getattr(transformers, 'CLIPModel', None)
                CLIPProcessor = getattr(transformers, 'CLIPProcessor', None)
                if CLIPModel and CLIPProcessor:
                    logger.info("Successfully imported CLIP via workaround method.")
                    return True
                else:
                    raise ImportError("CLIP classes not found in transformers")
            except Exception as e2:
                logger.error(f"CLIP workaround failed: {e2}")
                CLIPModel = None
                CLIPProcessor = None
                return False
        else:
            logger.warning(f"Unexpected CLIP import error: {e}")
            CLIPModel = None
            CLIPProcessor = None
            return False

# Attempt safe CLIP import
safe_import_clip()

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
        'accelerate': 'accelerate',
        'flash_attn': 'flash-attn (optional, for performance, install with --no-build-isolation)'
    }

    def __init__(self, model_path: str = None):
        # Use local model by default to avoid downloads
        default_local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "weights", "Qwen2.5-VL-3B-Instruct")
        
        if model_path is None:
            if os.path.exists(default_local_path):
                self.model_path = default_local_path
                logger.info(f"Using local Qwen model at: {self.model_path}")
            else:
                self.model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
                logger.warning(f"Local model not found, using remote: {self.model_path}")
        else:
            self.model_path = model_path
            logger.info(f"Using specified model path: {self.model_path}")
            
        self._check_dependencies()
        self.tokenizer = None 
        super().__init__()

    def _setup_model(self) -> None:
        """Setup the Qwen model."""
        super()._setup_model()

    def _get_model_name(self) -> str:
        """Get the model name for template system integration."""
        # Import ModelNames here to avoid circular imports
        try:
            from templates.template_manager import ModelNames
            return ModelNames.QWEN
        except ImportError:
            return "qwen"

    def analyze_image(self, image_path: str, quality: str = "standard", template_name: Optional[str] = None, 
                     template_variables: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str]]:
        """Analyze a single image with template support.
        
        Args:
            image_path: Path to the image file
            quality: Quality level - "standard", "detailed", or "creative"
            template_name: Specific template to use (e.g., "caption_detailed", "object_detection")
            template_variables: Variables for template substitution
            
        Returns:
            Tuple[str, Optional[str]]: (description, clean_caption)
        """
        return super().analyze_image(image_path, quality, template_name, template_variables)

    def analyze_images_batch(self, image_paths: List[str], quality: str = "standard",
                            template_name: Optional[str] = None, 
                            template_variables: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Optional[str]]]:
        """Analyze multiple images in batch with template support.
        
        Args:
            image_paths: List of paths to image files
            quality: Quality level - "standard", "detailed", or "creative"
            template_name: Specific template to use (e.g., "caption_detailed", "object_detection")
            template_variables: Variables for template substitution
            
        Returns:
            List[Tuple[str, Optional[str]]]: List of (description, clean_caption) tuples
        """
        return super().analyze_images_batch(image_paths, quality, template_name, template_variables)

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


class QwenCaptioner(BaseVisionModel):
    """Qwen2.5-VL-7B-Captioner-Relaxed model implementation optimized for detailed captions."""
    
    REQUIRED_PACKAGES = {
        'transformers': 'transformers (latest from git)',
        'torch': 'torch',
        'PIL': 'Pillow',
        'accelerate': 'accelerate',
        'bitsandbytes': 'bitsandbytes (for quantization)',
        'flash_attn': 'flash-attn (optional, for performance, install with --no-build-isolation)'
    }

    def __init__(self, model_path: str = None, use_quantization: str = None):
        # Use local 3B model by default to avoid downloads
        default_local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "weights", "Qwen2.5-VL-3B-Instruct")
        self.model_path = model_path or default_local_path
        
        # Validate local model exists
        if not model_path and not os.path.exists(self.model_path):
            logger.error(f"Local Qwen model not found at {self.model_path}")
            logger.error("Please download the model using: ./clone_models.sh")
            logger.error("Or place the Qwen2.5-VL-3B-Instruct model in models/weights/")
            raise FileNotFoundError(f"Qwen model not found at {self.model_path}")
        elif not model_path:
            logger.info(f"Using local Qwen2.5-VL-3B-Instruct model at: {self.model_path}")
            logger.info("Optimizing 3B model for captioning tasks with enhanced prompts")
        
        # Check for environment variable to force 4-bit quantization
        if os.getenv("QWEN_FORCE_4BIT", "").lower() in ["1", "true", "yes"]:
            self.use_quantization = "4bit"
            logger.info("Environment variable QWEN_FORCE_4BIT detected, forcing 4-bit quantization")
        else:
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
        """Get captioner-specific instructions optimized for best captioning results."""
        if quality == "standard":
            return "Create a clear, accurate caption for this image. Focus on the main subjects and their actions."
        elif quality == "detailed":
            return "Generate a comprehensive, detailed image caption. Describe all visible elements including: objects, people, animals, setting, colors, lighting, composition, mood, and any text. Be thorough but concise."
        elif quality == "creative":
            return "Write an engaging, descriptive caption that captures both the visual details and emotional essence of this image. Use vivid language to paint a picture with words."
        return "Write a descriptive caption for this image."

    def get_generation_params_captioner(self, quality: str) -> dict:
        """Get captioner-specific generation parameters optimized for best quality captioning."""
        if quality == "standard":
            return {
                "max_new_tokens": 80,
                "temperature": 0.3,  # Lower for more focused captions
                "top_p": 0.8,
                "do_sample": True,
                "repetition_penalty": 1.2
            }
        elif quality == "detailed":
            return {
                "max_new_tokens": 250,
                "temperature": 0.4,  # Slightly higher for detailed descriptions
                "top_p": 0.85,
                "repetition_penalty": 1.15,
                "do_sample": True,
                "num_beams": 2  # Reduced for faster performance
            }
        elif quality == "creative":
            return {
                "max_new_tokens": 180,
                "temperature": 0.7,  # Higher for creativity
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "top_k": 40
            }
        return {"max_new_tokens": 80, "temperature": 0.3, "top_p": 0.8, "do_sample": True}

    def _get_model_name(self) -> str:
        """Get the model name for template system integration."""
        # Import ModelNames here to avoid circular imports
        try:
            from templates.template_manager import ModelNames
            return ModelNames.QWEN
        except ImportError:
            return "qwen"

    def _get_legacy_prompt(self, quality: str) -> str:
        """Get legacy prompt for backward compatibility."""
        return self.get_instruction_for_quality_captioner(quality)

    def analyze_image(self, image_path: str, quality: str = "detailed", template_name: Optional[str] = None, 
                     template_variables: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str]]:
        """Override analyze_image to use captioner-specific methods with template support.
        
        Args:
            image_path: Path to the image file
            quality: Quality level - "standard", "detailed", or "creative" 
            template_name: Specific template to use (e.g., "caption_detailed", "object_detection")
            template_variables: Variables for template substitution
            
        Returns:
            Tuple[str, Optional[str]]: (description, clean_caption)
        """
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return "Error: Image file not found.", None
        
        try:
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Preprocess image for memory efficiency
            pil_image = self._preprocess_image_for_memory(pil_image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return "Error: Failed to load or process image.", None

        # Debug component availability for QwenCaptioner
        logger.debug(f"QwenCaptioner component status:")
        logger.debug(f"  _using_fallback: {getattr(self, '_using_fallback', False)}")
        logger.debug(f"  self.model: {self.model is not None}")
        logger.debug(f"  self.processor: {self.processor is not None}")
        logger.debug(f"  self.tokenizer: {self.tokenizer is not None}")
        logger.debug(f"  _QWEN_CLASS_AVAILABLE: {_QWEN_CLASS_AVAILABLE}")
        
        if getattr(self, '_using_fallback', False) or not all([self.model, self.processor, _QWEN_CLASS_AVAILABLE]):
            logger.warning("QwenCaptioner using fallback CLIP model for image analysis (Qwen components not fully available or in fallback mode).")
            logger.warning(f"Reason: _using_fallback={getattr(self, '_using_fallback', False)}, model={self.model is not None}, processor={self.processor is not None}, class_available={_QWEN_CLASS_AVAILABLE}")
            return self._analyze_with_clip(pil_image, quality)

        # Determine instruction/prompt to use
        if template_name is not None:
            # Get prompt from template
            instruction = self.get_prompt_from_template(quality, template_name, template_variables)
        else:
            # Use captioner-specific instruction for legacy mode
            instruction = self.get_instruction_for_quality_captioner(quality)
        
        # Enhanced system prompt for better captioning
        system_prompt = "You are an expert image captioning specialist. Create accurate, detailed, and engaging descriptions of images. Focus on what you can see clearly and describe it in natural, flowing language."
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": pil_image},
                ],
            },
        ]

        try:
            text_for_template = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = self.processor(
                text=[text_for_template],
                images=pil_image,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # Use captioner-specific generation parameters
            generation_params = self.get_generation_params_captioner(quality)
            
            # Use memory-efficient inference with mixed precision
            with torch.inference_mode():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    generated_ids = self.model.generate(**inputs, **generation_params)
            
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            caption = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )[0]
            caption = self.clean_output(caption)
            
            # Ensure caption is a string, not a list
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
            elif not isinstance(caption, str):
                caption = str(caption)
            
            model_name = "Qwen2.5-VL-3B-Instruct (Optimized for Captioning)"
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

            # Disable flash attention to prevent symbol conflicts
            logger.info("Flash Attention disabled to prevent symbol conflicts - using eager attention for stability")
            model_kwargs["attn_implementation"] = "eager"
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path, **model_kwargs)
            
            # Patch missing is_causal attribute for vision attention modules
            self._patch_vision_attention_is_causal()
            
            # Configure memory-efficient generation
            self._setup_memory_efficient_generation()
            
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

            
            # Aggressive memory cleanup before loading large model
            if torch.cuda.is_available():
                # Force garbage collection
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Multiple cleanup attempts for stubborn memory
                for _ in range(3):
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Check memory status after cleanup
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated()
                reserved_memory = torch.cuda.memory_reserved()
                free_memory = total_memory - reserved_memory  # Use reserved as baseline
                
                total_gb = total_memory / (1024**3)
                allocated_gb = allocated_memory / (1024**3)
                reserved_gb = reserved_memory / (1024**3)
                free_gb = free_memory / (1024**3)
                
                logger.info(f"GPU Memory Status After Cleanup - Total: {total_gb:.2f}GB, Allocated: {allocated_gb:.2f}GB, Reserved: {reserved_gb:.2f}GB, Free: {free_gb:.2f}GB")
                
                # With 7B model + very limited memory, force 4-bit quantization or CPU fallback
                if free_gb < 4.0:  # Critical: less than 4GB free, fallback to CPU CLIP
                    logger.error(f"Critical GPU memory shortage ({free_gb:.2f}GB free). Cannot load 7B model, falling back to CPU CLIP.")
                    self._load_clip_as_fallback_cpu(reason=f"Insufficient GPU memory ({free_gb:.2f}GB)")
                    return
                elif free_gb < 12.0:  # 7B model needs aggressive quantization
                    logger.warning(f"Limited GPU memory ({free_gb:.2f}GB). Forcing 4-bit quantization for 7B model.")
                    self.use_quantization = "4bit"
                elif free_gb < 16.0:  # Marginal memory, use 8-bit
                    logger.info(f"Moderate GPU memory ({free_gb:.2f}GB). Using 8-bit quantization.")
                    self.use_quantization = "8bit"
            
            logger.info(f"Loading Qwen2.5-VL-7B-Captioner-Relaxed model: {self.model_path}")
            
            model_kwargs: Dict[str, Any] = {
                "torch_dtype": torch.float16,  # Use FP16 for better memory efficiency
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,  # Enable low CPU memory usage
            }
            
            # Apply quantization if specified
            if self.use_quantization == "4bit":
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_storage=torch.bfloat16,
                )
                model_kwargs["quantization_config"] = bnb_config
                # Don't set torch_dtype when using quantization
                model_kwargs.pop("torch_dtype", None)
                logger.info("Using 4-bit quantization with bfloat16 for memory efficiency")
            elif self.use_quantization == "8bit":
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=False
                )
                model_kwargs["quantization_config"] = bnb_config
                # Don't set torch_dtype when using quantization
                model_kwargs.pop("torch_dtype", None)
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

            # Always use eager attention to prevent flash attention symbol conflicts
            model_kwargs["attn_implementation"] = "eager"
            logger.info("Using eager attention for stability and compatibility (flash attention disabled)")
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path, **model_kwargs)
            
            # Patch missing is_causal attribute for vision attention modules
            self._patch_vision_attention_is_causal()
            
            # Configure memory-efficient generation
            self._setup_memory_efficient_generation()
            
            logger.info(f"Successfully loaded QwenCaptioner model: {self.model_path} with kwargs: {model_kwargs}")
            
            # Set up processor with pixel limits for cost/quality balance
            min_pixels = 256*28*28
            max_pixels = 1280*28*28
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                max_pixels=max_pixels,
                min_pixels=min_pixels
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            logger.info(f"Successfully loaded QwenCaptioner processor and tokenizer for {self.model_path}")
            logger.info(f"Processor configured with min_pixels={min_pixels}, max_pixels={max_pixels}")

        except Exception as e:
            logger.error(f"Failed to initialize QwenCaptioner model: {str(e)}")
            # Clear cache and try with CPU fallback for CLIP
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._load_clip_as_fallback_cpu(reason=f"QwenCaptioner load failed: {e}")

    def _load_clip_as_fallback_cpu(self, reason: str) -> None:
        """Load CLIP model on CPU as fallback when GPU memory is insufficient."""
        logger.warning(f"Attempting to load CLIP model on CPU as fallback due to: {reason}")
        try:
            global CLIPModel, CLIPProcessor
            if CLIPModel is None or CLIPProcessor is None:
                from transformers import CLIPModel as DynamicCLIPModel, CLIPProcessor as DynamicCLIPProcessor
                CLIPModel = DynamicCLIPModel
                CLIPProcessor = DynamicCLIPProcessor
                if CLIPModel is None or CLIPProcessor is None:
                    raise ImportError("CLIPModel/CLIPProcessor not available for fallback.")

            fallback_model_id = "openai/clip-vit-base-patch32"
            logger.info(f"Loading fallback CLIP model on CPU: {fallback_model_id}")

            # Force CPU usage for fallback
            self.model = CLIPModel.from_pretrained(fallback_model_id, torch_dtype=torch.float32).to("cpu")
            self.processor = CLIPProcessor.from_pretrained(fallback_model_id)
            self.device = "cpu"  # Override device for fallback
            self.tokenizer = None 
            
            logger.info(f"Successfully loaded CLIP model as CPU fallback: {fallback_model_id}")
            self._using_fallback = True
        except Exception as fallback_error:
            logger.error(f"Failed to load CLIP CPU fallback model: {str(fallback_error)}")
            raise RuntimeError(f"QwenCaptioner model setup failed, and CPU fallback CLIP model also failed to load: {fallback_error}") from fallback_error

    def _get_legacy_prompt(self, quality: str) -> str:
        """Get legacy prompt for backward compatibility."""
        return self.get_instruction_for_quality(quality)

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

    def analyze_image(self, image_path: str, quality: str = "standard", template_name: Optional[str] = None, 
                     template_variables: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str]]:
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

        if getattr(self, '_using_fallback', False) or not all([self.model, self.processor, _QWEN_CLASS_AVAILABLE]):
            logger.info("Using fallback CLIP model for image analysis (Qwen components not fully available or in fallback mode).")
            return self._analyze_with_clip(pil_image, quality)

        # Determine instruction/prompt to use
        if template_name is not None:
            # Get prompt from template
            instruction = self.get_prompt_from_template(quality, template_name, template_variables)
        else:
            # Get quality-specific instruction for legacy mode
            instruction = self.get_instruction_for_quality(quality)
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert image describer."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": pil_image},
                ],
            },
        ]

        try:
            text_for_template = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = self.processor(
                text=[text_for_template],
                images=pil_image,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # Use quality-specific generation parameters
            generation_params = self.get_generation_params(quality)
            
            # Use memory-efficient inference with mixed precision
            with torch.inference_mode():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    generated_ids = self.model.generate(**inputs, **generation_params)
            
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            caption = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True  # Try setting to True to help with encoding issues
            )[0]
            caption = self.clean_output(caption)
            
            # Ensure caption is a string, not a list
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
            elif not isinstance(caption, str):
                caption = str(caption)
            
            model_name = "Qwen2.5-VL (non-AWQ)"
            description = f"Description: {caption}\n\nGenerated by: {model_name}"
            return description, caption

        except Exception as e:
            logger.error(f"Error generating caption with Qwen: {str(e)}")
            return self._analyze_with_clip(pil_image, quality)

    def analyze_images_batch(self, image_paths: List[str], quality: str = "standard",
                            template_name: Optional[str] = None, 
                            template_variables: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Optional[str]]]:
        if not image_paths:
            return []

        # Force single-image processing to prevent OOM
        logger.info(f"Processing {len(image_paths)} images individually to prevent OOM (RTX 4090 memory constraint)")
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.analyze_image(image_path, quality, template_name, template_variables)
                results.append(result)
                
                # Aggressive memory cleanup between images
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                logger.debug(f"Processed image {i+1}/{len(image_paths)}: {image_path}")
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {str(e)}")
                results.append((f"Error: Failed to process image {image_path}: {str(e)}", None))
                
                # Still cleanup memory even on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return results

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

    def _patch_vision_attention_is_causal(self):
        """Patch vision attention modules to add missing is_causal attribute."""
        try:
            # Define search paths in order of preference
            search_paths = []
            if hasattr(self.model, 'vision_model'):
                search_paths.append(self.model.vision_model)
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_model'):
                search_paths.append(self.model.model.vision_model)
            # Always include full model as fallback
            search_paths.append(self.model)
            
            patched_count = 0
            for search_root in search_paths:
                for name, module in search_root.named_modules():
                    if 'VisionAttention' in module.__class__.__name__ and not hasattr(module, 'is_causal'):
                        module.is_causal = False
                        logger.debug(f"Patched {name} with is_causal=False")
                        patched_count += 1
                if patched_count > 0:
                    break  # Stop after successful patching
                    
            if patched_count > 0:
                logger.info(f"Successfully patched {patched_count} vision attention modules with is_causal attribute")
            else:
                logger.warning("No vision attention modules found to patch")
                
        except Exception as e:
            logger.warning(f"Could not patch vision attention modules: {e}")

    def _preprocess_image_for_memory(self, image: Image.Image, max_resolution: int = 1024) -> Image.Image:
        """Resize image if too large to save memory.
        
        Args:
            image: PIL Image to potentially resize
            max_resolution: Maximum width or height allowed
            
        Returns:
            PIL Image, potentially resized
        """
        if isinstance(image, Image.Image):
            width, height = image.size
            if width > max_resolution or height > max_resolution:
                # Calculate new size maintaining aspect ratio
                ratio = min(max_resolution / width, max_resolution / height)
                new_size = (int(width * ratio), int(height * ratio))
                logger.info(f"Resizing image from {image.size} to {new_size} for memory efficiency")
                image = image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    def _setup_memory_efficient_generation(self):
        """Configure model for memory-efficient generation."""
        try:
            # Configure generation settings for memory efficiency
            if hasattr(self.model, 'generation_config'):
                current_max_tokens = getattr(self.model.generation_config, 'max_new_tokens', None)
                if current_max_tokens is not None:
                    self.model.generation_config.max_new_tokens = min(300, current_max_tokens)
                else:
                    self.model.generation_config.max_new_tokens = 300
                self.model.generation_config.num_beams = 1  # Disable beam search for memory
                if hasattr(self.model.generation_config, 'use_cache'):
                    self.model.generation_config.use_cache = False  # Disable KV cache
            
            # Enable gradient checkpointing if available
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing for memory efficiency")
            
            # Set memory fraction if CUDA available
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of GPU memory
                logger.info("Set CUDA memory fraction to 85%")
            
            # Enable memory efficient attention if available
            if hasattr(self.model, 'config'):
                if hasattr(self.model.config, 'use_memory_efficient_attention'):
                    self.model.config.use_memory_efficient_attention = True
                    
            logger.info("Memory-efficient generation settings configured")
        except Exception as e:
            logger.warning(f"Could not configure memory-efficient settings: {e}")

    @classmethod
    def is_available(cls) -> bool:
        if not _QWEN_CLASS_AVAILABLE:
            logger.warning("Qwen2_5_VLForConditionalGeneration class not found. Qwen model not available.")
            return False
        return True
