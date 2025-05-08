# models/qwen_model.py
from models.base_model import BaseVisionModel
import logging
import os
import sys
import platform
from typing import Tuple, Optional
import importlib

# Handle imports in a way that prevents circular dependencies
# Import only what's needed initially, defer other imports

# Safely import torch first
try:
    import torch
except ImportError as e:
    logging.error(f"PyTorch import error: {str(e)}")
    logging.error("PyTorch is required for all models. Please install it first.")

# Defer PIL import to when it's needed
try:
    from PIL import Image
except ImportError as e:
    logging.warning(f"PIL import error (will try again when needed): {str(e)}")

# Try transformers imports but handle failures gracefully
# Define variables that will be set properly by imports or remain None if imports fail
Qwen2_5_VLForConditionalGeneration = None
CLIPModel = None
CLIPProcessor = None

try:
    from transformers import AutoTokenizer, AutoProcessor
    
    # Try to import Qwen model class - this may fail with older transformers versions
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError as e:
        logging.warning(f"Qwen2_5_VLForConditionalGeneration import error: {str(e)}")
        logging.warning("This class requires the latest transformers: pip install git+https://github.com/huggingface/transformers.git")
    
    # Always import CLIP for fallback mechanism
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as e:
        logging.warning(f"CLIP model import error: {str(e)}")
        logging.warning("CLIP fallback won't be available; please install transformers")
        
except ImportError as e:
    logging.warning(f"Base transformers import error: {str(e)}")
    logging.warning("Make sure you have transformers installed: pip install git+https://github.com/huggingface/transformers.git")

logger = logging.getLogger(__name__)

class QwenModel(BaseVisionModel):
    """Qwen2.5-VL-3B-Instruct-AWQ model implementation for image captioning."""
    
    REQUIRED_PACKAGES = {
        'transformers': 'transformers',
        'torch': 'torch',
        'PIL': 'Pillow',
        'qwen_vl_utils': 'qwen-vl-utils[decord]==0.0.8'
    }

    def __init__(self, model_path: str = None):
        """
        Initialize the Qwen2.5-VL-3B-Instruct-AWQ model.
        
        Args:
            model_path (str): HuggingFace model path
        """
        # Try local path first, then fall back to HuggingFace
        if model_path is None:
            local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "weights", "Qwen2.5-VL-3B-Instruct-AWQ")
            if os.path.exists(local_path):
                self.model_path = local_path
                logger.info(f"Using local model: {self.model_path}")
            else:
                self.model_path = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
                logger.info(f"Using remote model: {self.model_path}")
        else:
            self.model_path = model_path
            logger.info(f"Using specified model: {self.model_path}")
            
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
        """Set up the Qwen2.5-VL model with enhanced error handling and fallback options."""
        try:
            # Check if we have the required Qwen model class 
            if Qwen2_5_VLForConditionalGeneration is None:
                # Try one more time to import it directly here
                try:
                    logger.info("Attempting direct import of Qwen2_5_VLForConditionalGeneration...")
                    from transformers import Qwen2_5_VLForConditionalGeneration as DirectQwenClass
                    # Make it available to global scope
                    global Qwen2_5_VLForConditionalGeneration
                    Qwen2_5_VLForConditionalGeneration = DirectQwenClass
                    logger.info("Successfully imported Qwen2_5_VLForConditionalGeneration on demand")
                except ImportError as e:
                    logger.error(f"Cannot import Qwen2_5_VLForConditionalGeneration: {str(e)}")
                    logger.error("Please upgrade transformers: pip install git+https://github.com/huggingface/transformers.git")
                    raise ImportError("Missing Qwen2_5_VLForConditionalGeneration class - please upgrade transformers") from e
                    
            # Try to import qwen_vl_utils for vision processing
            try:
                from qwen_vl_utils import process_vision_info
                self.process_vision_info = process_vision_info
            except ImportError as e:
                logger.warning(f"Missing qwen_vl_utils: {str(e)}")
                logger.warning("Please install: pip install qwen-vl-utils[decord]==0.0.8")
                self.process_vision_info = None
            
            logger.info(f"Loading Qwen2.5-VL model from {self.model_path}...")
            
            # Try to determine if this is an AWQ model or a standard model based on the path
            is_awq_model = "AWQ" in self.model_path or "awq" in self.model_path
            
            try:
                # First attempt: Try with specific Qwen2_5_VLForConditionalGeneration class
                logger.info("Trying to load with Qwen2_5_VLForConditionalGeneration...")
                
                # Start with basic configuration for model loading
                model_kwargs = {
                    "torch_dtype": self.torch_dtype,
                    "device_map": "auto",
                    "trust_remote_code": True
                }
                
                # Check for low memory situation and apply optimizations if needed
                if hasattr(self, 'get_low_memory_optimization_settings'):
                    mem_settings = self.get_low_memory_optimization_settings()
                    logger.info(f"Applying memory optimization settings for device: {self.device}")
                    model_kwargs.update(mem_settings)
                
                # Only set quantization config if it's an AWQ model and we're not loading a local path
                if is_awq_model and not os.path.isdir(self.model_path):
                    try:
                        from transformers import AwqConfig
                        logger.info("Setting AWQ quantization configuration...")
                        model_kwargs["quantization_config"] = AwqConfig(bits=4)
                    except (ImportError, Exception) as qe:
                        logger.warning(f"Could not set up AWQ config: {str(qe)}")
                        logger.warning("For AWQ support: pip install 'autoawq>=0.1.8'")
                
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
                logger.info("Successfully loaded Qwen2.5-VL model")
                
                # Load processor with trust_remote_code
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                logger.info("Successfully loaded Qwen2.5-VL processor")
                
            except Exception as first_error:
                logger.warning(f"First loading attempt failed: {str(first_error)}")
                logger.info("Trying alternative loading approach...")
                
                try:
                    # Second attempt: Try without AWQ quantization
                    if is_awq_model:
                        logger.info("Attempting to load non-AWQ version...")
                        
                        # Try a non-AWQ version of the model
                        alternative_model_path = self.model_path.replace("-AWQ", "").replace("awq", "")
                        if alternative_model_path == self.model_path:
                            alternative_model_path = "Qwen/Qwen2.5-VL-3B-Instruct"  # Fallback to non-AWQ
                        
                        logger.info(f"Trying alternative model: {alternative_model_path}")
                        
                        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                            alternative_model_path,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto",
                            trust_remote_code=True
                        )
                        
                        self.processor = AutoProcessor.from_pretrained(
                            alternative_model_path,
                            trust_remote_code=True
                        )
                        logger.info(f"Successfully loaded alternative Qwen2.5-VL model: {alternative_model_path}")
                    else:
                        # If it's not an AWQ issue, re-raise the original error
                        raise first_error
                    
                except Exception as second_error:
                    logger.error(f"Failed to load model with alternative approach: {str(second_error)}")
                    
                    # Third attempt: Try with CLIP model as fallback
                    logger.warning("Attempting to load CLIP model as a fallback...")
                    try:
                        from transformers import CLIPProcessor, CLIPModel
                        
                        fallback_model = "openai/clip-vit-base-patch32"
                        self.model = CLIPModel.from_pretrained(fallback_model)
                        self.processor = CLIPProcessor.from_pretrained(fallback_model)
                        logger.info(f"Loaded CLIP model as fallback: {fallback_model}")
                        
                        # This will be used to identify that we're using a fallback model
                        self._using_fallback = True
                        
                    except Exception as fallback_error:
                        logger.error(f"Failed to load fallback model: {str(fallback_error)}")
                        raise RuntimeError(f"All model loading attempts failed. Original error: {str(first_error)}") from first_error
            
            # Store the process_vision_info function for later use
            self.process_vision_info = process_vision_info
            self._using_fallback = getattr(self, '_using_fallback', False)
                
        except Exception as e:
            error_msg = f"Failed to initialize Qwen2.5-VL model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def analyze_image(self, image_path: str, quality: str = "standard") -> Tuple[str, Optional[str]]:
        """
        Analyze an image using the Qwen2.5-VL model or fallback.
        
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
            
            # Check if we're using a fallback model (CLIP)
            if getattr(self, '_using_fallback', False):
                logger.info("Using fallback CLIP model for image analysis")
                return self._analyze_with_clip(image, quality)
            
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
                        {"type": "image", "image": image},  # Use image object directly
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
                    try:
                        # Normal processing with processor and process_vision_info
                        if hasattr(self, 'process_vision_info'):
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
                        else:
                            # Fallback for when process_vision_info is not available
                            inputs = self.processor(
                                images=image,
                                text=instruction,
                                return_tensors="pt"
                            )
                        
                        inputs = inputs.to(self.device)
                    except Exception as processor_error:
                        logger.error(f"Error using processor: {processor_error}")
                        # Try the fallback CLIP model approach
                        return self._analyze_with_clip(image, quality)
                    
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
                    
                    try:
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p
                        )
                    except Exception as gen_error:
                        logger.error(f"Error during generation: {str(gen_error)}")
                        # Try without keyword arguments if the model interface is different
                        try:
                            if "input_ids" in inputs and "pixel_values" in inputs:
                                # Use specific arguments pattern
                                generated_ids = self.model.generate(
                                    input_ids=inputs["input_ids"],
                                    pixel_values=inputs["pixel_values"],
                                    max_new_tokens=max_new_tokens,
                                    do_sample=True,
                                    temperature=temperature,
                                    top_p=top_p
                                )
                            else:
                                # Generic generation
                                generated_ids = self.model.generate(
                                    inputs["input_ids"],
                                    max_new_tokens=max_new_tokens,
                                    do_sample=True
                                )
                        except Exception as fallback_gen_error:
                            logger.error(f"Fallback generation failed: {str(fallback_gen_error)}")
                            return self._analyze_with_clip(image, quality)
                
                # Extract and decode the generated tokens
                try:
                    if "input_ids" in inputs:
                        # Extract only the newly generated tokens
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                        ]
                    else:
                        # Just use all the generated tokens
                        generated_ids_trimmed = generated_ids
                    
                    # Decode caption
                    if self.processor is not None and hasattr(self.processor, 'batch_decode'):
                        caption = self.processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                    elif hasattr(self.model, 'decode') and callable(getattr(self.model, 'decode')):
                        caption = self.model.decode(generated_ids_trimmed[0])
                    else:
                        # Fallback to tokenizer
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                        caption = tokenizer.decode(generated_ids_trimmed[0], skip_special_tokens=True)
                except Exception as decode_error:
                    logger.error(f"Error decoding output: {str(decode_error)}")
                    return self._analyze_with_clip(image, quality)
                
                # Clean output
                caption = self.clean_output(caption)
                
                # Prepare description with the model name
                model_name = "Qwen2.5-VL"
                if getattr(self, '_using_fallback', False):
                    model_name += " (Fallback Mode)"
                description = f"Description: {caption}\n\nGenerated by: {model_name}"
                
                return description, caption
                
            except Exception as e:
                logger.error(f"Error generating caption: {str(e)}")
                # Try the fallback method
                return self._analyze_with_clip(image, quality)
                
        except Exception as e:
            logger.error(f"Error analyzing image with Qwen2.5-VL: {str(e)}")
            return f"Error: An unexpected error occurred. {str(e)}", None
            
    def _analyze_with_clip(self, image, quality: str = "standard") -> Tuple[str, Optional[str]]:
        """Fallback method using CLIP model for image analysis."""
        try:
            logger.info("Using CLIP fallback for image analysis")
            
            # First make sure we have CLIPModel/CLIPProcessor classes available
            if CLIPModel is None or CLIPProcessor is None:
                # Try importing again at runtime if not available from module import
                try:
                    from transformers import CLIPProcessor, CLIPModel as ImportedCLIPModel
                    global CLIPModel, CLIPProcessor
                    CLIPModel = ImportedCLIPModel  # Make available to global scope
                    logger.info("Successfully imported CLIPModel and CLIPProcessor on demand")
                except ImportError as e:
                    logger.error(f"Failed to import CLIP models for fallback: {str(e)}")
                    return (f"Error: Cannot use CLIP fallback mode. Please install or update transformers: {str(e)}", None)
            
            # Check if we already have a CLIP model loaded
            need_new_model = True
            if hasattr(self, 'model') and self.model is not None:
                # Check if it's a CLIP model - use string comparison instead of isinstance
                model_class_name = self.model.__class__.__name__
                if model_class_name == "CLIPModel":
                    need_new_model = False
                    logger.info("Using existing CLIP model")
            
            # Load a new CLIP model if needed
            if need_new_model:
                fallback_model_id = "openai/clip-vit-base-patch32"
                logger.info(f"Loading fallback CLIP model: {fallback_model_id}")
                self.model = CLIPModel.from_pretrained(fallback_model_id)
                self.processor = CLIPProcessor.from_pretrained(fallback_model_id)
                # Move the fallback model to the correct device
                self.model.to(self.device) 
                logger.info(f"CLIP model moved to device: {self.device}")
            
            # Prepare image with processor
            inputs = self.processor(
                text=["a photo of a landscape", "a portrait", "a photo of food", 
                      "a photo of an animal", "a photo of a building", "a photo of people"],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device) # Ensure inputs are also on the correct device
            
            # Get prediction
            with torch.inference_mode():
                outputs = self.model(**inputs)
                
            # Get the image-text similarity scores
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).tolist()[0]
            
            # Get the most likely category
            categories = ["landscape", "portrait", "food", "animal", "building", "people"]
            scores = list(zip(categories, probs))
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Generate a caption based on quality level
            top_category = scores[0][0]
            confidence = scores[0][1] * 100
            
            if quality == "detailed":
                # More detailed description
                other_elements = [f"{cat} ({conf:.1f}%)" for cat, conf in scores[1:3]]
                caption = f"This image appears to be a {top_category} (confidence: {confidence:.1f}%). "
                caption += f"It may also contain elements of {' and '.join(other_elements)}."
            elif quality == "creative":
                # More creative description
                caption = f"An interesting {top_category} scene captured in this image. "
                if top_category == "landscape":
                    caption += "The natural beauty unfolds across the frame, inviting the viewer to explore its details."
                elif top_category == "portrait":
                    caption += "The subject's presence dominates the composition, telling a story through expression and posture."
                elif top_category == "food":
                    caption += "The culinary delight is presented with care, enticing the viewer's appetite."
                elif top_category == "animal":
                    caption += "The creature's character and form create a compelling focal point in the image."
                elif top_category == "building":
                    caption += "The architectural elements reveal both function and aesthetic considerations."
                else:  # people
                    caption += "The human elements bring life and scale to the composition."
            else:  # standard
                caption = f"This image shows a {top_category}."
                
            description = f"Description: {caption}\n\nGenerated by: CLIP (Fallback Mode)"
            return description, caption
            
        except Exception as e:
            logger.error(f"Error in CLIP fallback analysis: {str(e)}")
            return f"Image analysis failed with both primary and fallback models. Error: {str(e)}", None

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
            # Check for Qwen2_5_VLForConditionalGeneration support
            import transformers
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                has_qwen_class = True
            except ImportError:
                logger.error("Your transformers version doesn't support Qwen2_5_VLForConditionalGeneration. Please install transformers==4.36.2.")
                has_qwen_class = False
            
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
            
            return has_qwen_class and has_qwen_utils
        except ImportError as e:
            logger.error(f"Required package not found: {str(e)}")
            return False