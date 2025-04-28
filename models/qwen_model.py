# models/qwen_model.py
from models.base_model import BaseVisionModel
import logging
import torch
import importlib
from PIL import Image
import os
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM
from transformers.processing_utils import ProcessingMixin

# Try importing specific processor class, but don't fail if not available
try:
    from transformers import Qwen2VLProcessor
    HAS_SPECIFIC_PROCESSOR = True
except ImportError:
    HAS_SPECIFIC_PROCESSOR = False
    from transformers import AutoProcessor

logger = logging.getLogger(__name__)

class QwenModel(BaseVisionModel):
    """Qwen2.5-VL-3B-Instruct-AWQ model implementation for image captioning."""
    
    REQUIRED_PACKAGES = {
        'transformers': 'transformers',
        'torch': 'torch',
        'PIL': 'Pillow',
        'qwen_vl_utils': 'qwen-vl-utils[decord]==0.0.8'
    }

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"):
        """
        Initialize the Qwen2.5-VL-3B-Instruct-AWQ model.
        
        Args:
            model_path (str): HuggingFace model path
        """
        self.model_path = model_path
        self._check_dependencies()
        super().__init__()

    @classmethod
    def _check_dependencies(cls) -> None:
        """Check for required dependencies and provide clear installation instructions."""
        missing_packages = []
        for package, pip_name in cls.REQUIRED_PACKAGES.items():
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append((package, pip_name))

        if missing_packages:
            install_command = "pip install " + " ".join(pkg[1] for pkg in missing_packages)
            error_msg = (
                f"Missing required packages: {', '.join(pkg[0] for pkg in missing_packages)}\n"
                f"Please install them using:\n{install_command}\n\n"
                f"It's highly recommended to use `qwen-vl-utils[decord]` feature for faster video loading."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _setup_model(self) -> None:
        """Set up the Qwen2.5-VL-3B-Instruct-AWQ model."""
        try:
            from qwen_vl_utils import process_vision_info
            
            logger.info(f"Loading Qwen2.5-VL model from {self.model_path}...")
            
            # Initialize model with trust_remote_code=True which is essential
            try:
                # Load the model first as it's more likely to succeed
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,  # This is critical for newer models
                    device_map="auto" if torch.cuda.is_available() else None
                )
                logger.info("Successfully loaded Qwen2.5-VL model")
                
                # Try to load processor with specific class first, then fall back
                try:
                    if HAS_SPECIFIC_PROCESSOR:
                        logger.info("Attempting to load with specific Qwen2VLProcessor...")
                        self.processor = Qwen2VLProcessor.from_pretrained(self.model_path)
                    else:
                        logger.info("Specific processor not available, using AutoProcessor...")
                        self.processor = AutoProcessor.from_pretrained(
                            self.model_path, 
                            trust_remote_code=True
                        )
                    logger.info("Successfully loaded Qwen2.5-VL processor")
                except Exception as proc_error:
                    logger.warning(f"Error loading processor: {str(proc_error)}")
                    logger.info("Attempting direct model usage without processor...")
                    # We'll handle operations that need the processor in analyze_image
                    self.processor = None
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise RuntimeError("Model initialization failed. Make sure transformers library is updated.") from e
            
            # Store the process_vision_info function for later use
            self.process_vision_info = process_vision_info
                
        except Exception as e:
            error_msg = f"Failed to initialize Qwen2.5-VL model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def analyze_image(self, image_path: str, quality: str = "standard") -> Tuple[str, Optional[str]]:
        """
        Analyze an image using the Qwen2.5-VL model.
        
        Args:
            image_path (str): Path to the image file
            quality (str): Quality level - "standard", "detailed", or "creative"
            
        Returns:
            Tuple[str, Optional[str]]: (description, clean_caption)
        """
        try:
            # Validate image path
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return "Error: Image file not found.", None
                
            # Load and validate image
            try:
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {str(e)}")
                return "Error: Failed to load or process image.", None
            
            # Prepare prompt based on quality
            if quality == "detailed":
                instruction = "Describe this image in detail, including all subjects, actions, background elements, colors, and visual composition."
            elif quality == "creative":
                instruction = "Describe this image in a creative and vivid way, focusing on artistic elements, mood, and storytelling aspects."
            else:  # standard
                instruction = "Describe this image."

            # Create messages structure for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]

            # Prepare inputs for the model
            try:
                if self.processor is None:
                    logger.warning("No processor available. Using direct model access approach.")
                    # This is a simplified approach when the processor isn't available
                    # It might not work for all cases but it's a fallback
                    from transformers import AutoTokenizer
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                        # Simple tokenization of the instruction
                        text = f"<image>\n{instruction}"
                        inputs = tokenizer(text, return_tensors="pt").to(self.device)
                        # Add the image as pixel_values (simplified approach)
                        inputs["pixel_values"] = torch.stack([self._preprocess_image(image)]).to(self.device)
                    except Exception as tokenizer_error:
                        logger.error(f"Failed to create inputs without processor: {str(tokenizer_error)}")
                        return "Error: Failed to process inputs. The model processor is not available.", None
                else:
                    # Normal processing with processor
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = self.process_vision_info(messages)
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(self.device)
                    
                # Add support for handling processor-less generation
                
                # Generate caption
                with torch.inference_mode():
                    if quality == "detailed":
                        max_new_tokens = 200
                        temperature = 0.7
                        top_p = 0.9
                    elif quality == "creative":
                        max_new_tokens = 150
                        temperature = 0.9
                        top_p = 0.95
                    else:  # standard
                        max_new_tokens = 128
                        temperature = 0.7
                        top_p = 0.9
                    
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p
                    )
                
                # Extract and decode the generated tokens
                if self.processor is None:
                    try:
                        # Fallback to using tokenizer for decoding
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                        
                        # Extract only the newly generated tokens
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        
                        # Decode caption using tokenizer instead of processor
                        caption = tokenizer.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                    except Exception as decode_error:
                        logger.error(f"Error decoding output: {str(decode_error)}")
                        caption = "Error decoding model output. The processor is not available."
                else:
                    # Extract only the newly generated tokens
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    
                    # Decode caption using processor
                    caption = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                
                # Clean output
                caption = self.clean_output(caption)
                
                # Prepare description with the model name
                description = f"Description: {caption}\n\nGenerated by: Qwen2.5-VL-3B-Instruct-AWQ"
                
                return description, caption
                
            except Exception as e:
                logger.error(f"Error generating caption: {str(e)}")
                return f"Error: Failed to generate image description. {str(e)}", None
                
        except Exception as e:
            logger.error(f"Error analyzing image with Qwen2.5-VL: {str(e)}")
            return f"Error: An unexpected error occurred. {str(e)}", None

    def _preprocess_image(self, image):
        """
        Preprocess an image for the model when no processor is available.
        
        Args:
            image (PIL.Image): The input image
            
        Returns:
            torch.Tensor: The preprocessed image tensor
        """
        try:
            # Basic preprocessing for vision models (resize to 224x224, normalize)
            from torchvision import transforms
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            return transform(image)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            # Return a blank tensor as fallback
            return torch.zeros((3, 224, 224), device=self.device)
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if the model can be initialized with current environment."""
        try:
            # Check for transformers with AutoModel support
            import transformers
            try:
                from transformers import AutoModelForCausalLM
                has_transformer = True
            except ImportError:
                logger.error("Your transformers version doesn't support AutoModelForCausalLM. Please update transformers.")
                has_transformer = False
            
            # Check for qwen_vl_utils
            try:
                import qwen_vl_utils
                has_qwen_utils = True
            except ImportError:
                logger.error("qwen_vl_utils not found. Please install: pip install qwen-vl-utils[decord]==0.0.8")
                has_qwen_utils = False
            
            # Check CUDA availability
            import torch
            has_cuda = torch.cuda.is_available()
            if not has_cuda:
                logger.warning("CUDA not available. Model will run in CPU mode with greatly reduced performance.")
            
            return has_transformer and has_qwen_utils
        except ImportError as e:
            logger.error(f"Required package not found: {str(e)}")
            return False