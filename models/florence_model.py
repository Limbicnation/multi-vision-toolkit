# models/florence_model.py
from models.base_model import BaseVisionModel
from typing import Tuple, Optional, List
import logging
import torch
import importlib
from packaging.version import parse as parse_version
from PIL import Image
import os
import sys
import platform

# Handle imports in a way that prevents circular dependencies
# Import only what's needed initially, defer other imports
import importlib
import logging

# Safely check for torch/torchvision versions without direct imports
def _check_installed_version(package_name):
    try:
        if importlib.util.find_spec(package_name) is None:
            return None
        
        package = importlib.import_module(package_name)
        if hasattr(package, '__version__'):
            return package.__version__
        return "Unknown version"
    except Exception as e:
        logging.warning(f"Error checking {package_name} version: {str(e)}")
        return None

# Check versions without full imports
torch_version = _check_installed_version('torch')
torchvision_version = _check_installed_version('torchvision')

# Log basic version info
logging.info(f"PyTorch version: {torch_version}")
logging.info(f"Torchvision version: {torchvision_version}")

# Import torch, but defer torchvision import to when it's actually needed
try:
    import torch
except ImportError as e:
    logging.error(f"PyTorch import error: {str(e)}")
    logging.error("PyTorch is required for all models. Please install it first.")
    # We don't raise here, as the error will be handled in dependency check

logger = logging.getLogger(__name__)

class Florence2Model(BaseVisionModel):
    """Florence-2 vision model implementation following official guidelines."""
    
    REQUIRED_PACKAGES = {
        'transformers': 'transformers',
        'timm': 'timm==0.9.12', # Enforce specific timm version
        'einops': 'einops',
        'torch': 'torch', # General check, PyTorch version check is also in _setup_model
        'torchvision': 'torchvision>=0.17.0',
        'PIL': 'Pillow',
        'packaging': 'packaging' # For version parsing
    }

    def __init__(self):
        """Initialize Florence-2 model with dependency checks."""
        self._check_dependencies()
        super().__init__()

    @classmethod
    def _check_dependencies(cls) -> None:
        """Check for required dependencies and their versions, providing clear installation instructions."""
        missing_packages_info = [] # Stores tuples of (descriptive_error, pip_install_spec)
        
        for package_name, pip_name_spec in cls.REQUIRED_PACKAGES.items():
            try:
                module = importlib.import_module(package_name)
                installed_version_str = getattr(module, '__version__', None)

                if '==' in pip_name_spec:
                    _, required_version_str = pip_name_spec.split('==')
                    if not installed_version_str:
                        msg = f"{package_name} (version {required_version_str} required, but installed version unknown)"
                        logger.warning(msg)
                        missing_packages_info.append((msg, pip_name_spec))
                    elif parse_version(installed_version_str) != parse_version(required_version_str):
                        msg = f"{package_name} (installed: {installed_version_str}, required: {required_version_str})"
                        logger.warning(msg)
                        missing_packages_info.append((msg, pip_name_spec))
                
                elif '>=' in pip_name_spec:
                    _, required_version_str = pip_name_spec.split('>=')
                    if not installed_version_str:
                        msg = f"{package_name} (version >= {required_version_str} required, but installed version unknown)"
                        logger.warning(msg)
                        missing_packages_info.append((msg, pip_name_spec))
                    elif parse_version(installed_version_str) < parse_version(required_version_str):
                        msg = f"{package_name} (installed: {installed_version_str}, required: >= {required_version_str})"
                        logger.warning(msg)
                        missing_packages_info.append((msg, pip_name_spec))
                
                # Special handling for torchvision import robustness if needed, though import_module and version check should be primary
                if package_name == 'torchvision' and installed_version_str:
                    # Example: torchvision.__version__ might be checked again or specific attributes
                    # For now, the generic version check is assumed sufficient.
                    pass

            except ImportError:
                msg = f"{package_name} (module not found, required: {pip_name_spec})"
                logger.warning(msg)
                missing_packages_info.append((msg, pip_name_spec))
            except Exception as e:
                msg = f"{package_name} (error during check: {str(e)}, required: {pip_name_spec})"
                logger.error(msg)
                missing_packages_info.append((msg, pip_name_spec))

        if missing_packages_info:
            # Ensure unique pip_name_spec for the install command
            unique_install_specs = sorted(list(set(spec for _, spec in missing_packages_info)))
            install_command = "pip install " + " ".join(unique_install_specs)
            
            error_details = [desc for desc, _ in missing_packages_info]
            error_msg_summary = (
                f"Missing or incorrect versions for required packages:\n"
                + "\n".join([f"  - {detail}" for detail in error_details]) +
                f"\n\nPlease install or update them using:\n{install_command}"
            )
            logger.error(error_msg_summary)
            raise RuntimeError(error_msg_summary)

    def _setup_model(self) -> None:
        """Set up the Florence-2 model following official implementation."""
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch # Ensure torch is imported within the method scope as well
        
        logger.info("Loading Florence-2 model...")
        hub_model_path = "microsoft/Florence-2-base"
        local_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "weights", "Florence-2-base")
        
        model_path_to_use = None
        model_loaded_successfully = False
        last_error = None
        errors_encountered = [] # To store all errors

        # Check PyTorch version for vulnerability warning (do this once)
        current_torch_version_str = torch.__version__
        required_torch_version_str = "2.6.0" # Example, adjust as needed
        try:
            current_torch_ver_parts = list(map(int, current_torch_version_str.split('+')[0].split('.')))
            required_torch_ver_parts = list(map(int, required_torch_version_str.split('.')))
            if current_torch_ver_parts < required_torch_ver_parts:
                logger.warning(
                    f"Your PyTorch version ({current_torch_version_str}) is older than the recommended minimum ({required_torch_version_str}). "
                    f"Consider upgrading PyTorch for security and performance improvements. "
                    f"You can typically upgrade using: pip install --upgrade torch torchvision"
                )
        except Exception as e_ver:
            logger.debug(f"Failed to parse PyTorch version for comparison: {str(e_ver)}")

        # Ensure necessary libraries are imported for model loading
        if 'torchvision' not in sys.modules:
            logger.info("Importing torchvision for Florence-2 model setup.")
            try:
                import torchvision
            except Exception as tv_error:
                logger.warning(f"Torchvision import error (non-critical during setup): {str(tv_error)}")
        if 'timm' not in sys.modules:
            logger.info("Importing timm for Florence-2 model setup.")
            try:
                import timm
            except Exception as timm_error:
                logger.warning(f"Timm import error (non-critical during setup): {str(timm_error)}")
        
        # Attempt 1: Load from Hub with safetensors
        logger.info(f"Attempt 1: Loading model from Hub: {hub_model_path} with safetensors...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                hub_model_path,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                device_map="auto", # Let transformers handle device mapping
                revision="main",
                local_files_only=False 
            )
            model_loaded_successfully = True
            model_path_to_use = hub_model_path
            logger.info(f"Successfully loaded model from Hub using safetensors: {hub_model_path}")
        except Exception as e_hub_sft:
            last_error = e_hub_sft
            errors_encountered.append(f"Hub (safetensors): {str(e_hub_sft)}")
            logger.warning(f"Attempt 1 (Hub, safetensors) failed: {str(e_hub_sft)}")

            # Attempt 2: Load from Hub with standard method (pickle)
            logger.info(f"Attempt 2: Loading model from Hub: {hub_model_path} with standard method...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    hub_model_path,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    local_files_only=False, # Ensure this is False for Hub
                    revision="main"
                ).to(self.device) # Explicitly move to device if not using device_map
                model_loaded_successfully = True
                model_path_to_use = hub_model_path
                logger.info(f"Successfully loaded model from Hub using standard method: {hub_model_path}")
            except Exception as e_hub_std:
                last_error = e_hub_std
                errors_encountered.append(f"Hub (standard): {str(e_hub_std)}")
                logger.warning(f"Attempt 2 (Hub, standard) failed: {str(e_hub_std)}")

        # If Hub loading failed, try local path if it exists
        if not model_loaded_successfully:
            if os.path.exists(local_model_path):
                logger.info(f"Hub loading failed. Attempting fallback to local path: {local_model_path}")
                # Attempt 3: Load from local with safetensors
                logger.info(f"Attempt 3: Loading local model {local_model_path} with safetensors...")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        local_model_path,
                        torch_dtype=self.torch_dtype,
                        trust_remote_code=True,
                        use_safetensors=True,
                        low_cpu_mem_usage=True,
                        device_map="auto", 
                        revision="main", # revision might not be relevant for local_files_only
                        local_files_only=True
                    )
                    model_loaded_successfully = True
                    model_path_to_use = local_model_path
                    logger.info(f"Successfully loaded model from local path using safetensors: {local_model_path}")
                except Exception as e_local_sft:
                    last_error = e_local_sft
                    errors_encountered.append(f"Local (safetensors, path: {local_model_path}): {str(e_local_sft)}")
                    logger.warning(f"Attempt 3 (Local, safetensors) failed: {str(e_local_sft)}")

                    # Attempt 4: Load from local with standard method
                    logger.info(f"Attempt 4: Loading local model {local_model_path} with standard method...")
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            local_model_path,
                            torch_dtype=self.torch_dtype,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            local_files_only=True,
                            revision="main" 
                        ).to(self.device)
                        model_loaded_successfully = True
                        model_path_to_use = local_model_path
                        logger.info(f"Successfully loaded model from local path using standard method: {local_model_path}")
                    except Exception as e_local_std:
                        last_error = e_local_std
                        errors_encountered.append(f"Local (standard, path: {local_model_path}): {str(e_local_std)}")
                        logger.warning(f"Attempt 4 (Local, standard) failed: {str(e_local_std)}")
            else: # Hub failed and no local path
                 logger.warning(f"Hub loading failed and local model path '{local_model_path}' not found. Cannot attempt local loading.")
                 # Add a placeholder error if last_error is still from Hub attempts
                 if not any("Local" in err for err in errors_encountered):
                     errors_encountered.append(f"Local: Path '{local_model_path}' does not exist.")


        if not model_loaded_successfully:
            logger.error(f"All attempts to load Florence-2 model failed.")
            error_summary = "\n".join([f"  - {err}" for err in errors_encountered])
            logger.error(f"Error summary:\n{error_summary}")
            
            error_str_context = str(last_error).lower() if last_error else ""
            
            if "safetensors" in error_str_context: extra_package = "pip install safetensors"
            elif "timm" in error_str_context or "davit" in error_str_context: extra_package = "pip install timm==0.9.12" 
            elif "einops" in error_str_context: extra_package = "pip install einops"
            else: extra_package = "pip install timm==0.9.12 safetensors einops" # General recommendation
            
            # Platform-specific installation commands
            try:
                is_wsl = sys.platform == "linux" and "microsoft" in os.uname().release.lower()
            except AttributeError:
                # os.uname() not available on Windows
                is_wsl = False
            
            if sys.platform == "win32" or is_wsl: # Windows or WSL check
                torch_install = "pip install torch==2.6.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121" # Assuming CUDA 12.1
            elif sys.platform == "darwin" and platform.processor() == "arm": # Apple Silicon
                torch_install = "pip install torch==2.6.0 torchvision==0.17.0" # MPS support
            else: # Generic Linux with CUDA
                torch_install = "pip install torch==2.6.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121"
            
            error_message_guidance = (
                f"Model loading failed. This may be due to a PyTorch security restriction, missing dependencies, or incorrect model files.\n\n"
                f"Review the error summary above. Common solutions:\n"
                f"1. Ensure PyTorch and Torchvision are correctly installed for your system:\n   {torch_install}\n"
                f"2. Install or update other common dependencies:\n   {extra_package}\n"
                f"3. Upgrade transformers and accelerate:\n   pip install --upgrade transformers accelerate\n"
                f"4. If loading from local path, verify the model files at '{local_model_path}' are complete and not corrupted.\n"
                f"5. If loading from Hub, check your internet connection and Hugging Face Hub status.\n\n"
                f"Last specific error encountered: {str(last_error)}"
            )
            raise RuntimeError(f"Model initialization failed. {error_message_guidance}") from last_error

        # Load processor
        processor_load_path = model_path_to_use # Use the path that successfully loaded the model
        logger.info(f"Loading processor using path: {processor_load_path}")
        try:
            self.processor = AutoProcessor.from_pretrained(
                processor_load_path,
                trust_remote_code=True,
                local_files_only=(processor_load_path == local_model_path and os.path.exists(local_model_path)) 
            )
            logger.info(f"Successfully loaded processor from: {processor_load_path}")
        except Exception as e_proc:
            logger.error(f"Failed to load processor from {processor_load_path}: {str(e_proc)}")
            # Try with explicit Hub path as a last resort for processor if a local path was used for model
            if processor_load_path == local_model_path and hub_model_path != local_model_path:
                logger.info(f"Attempting to load processor from explicit Hub path: {hub_model_path}")
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        hub_model_path,
                        trust_remote_code=True,
                        local_files_only=False # Explicitly Hub
                    )
                    logger.info(f"Successfully loaded processor from Hub path: {hub_model_path}")
                except Exception as e_hub_proc:
                     logger.error(f"Failed to load processor from Hub path {hub_model_path} as well: {str(e_hub_proc)}")
                     raise RuntimeError(f"Processor initialization failed from both '{processor_load_path}' and '{hub_model_path}'. Last Hub error: {str(e_hub_proc)}") from e_hub_proc
            else: # If Hub path was already tried for processor or no other option
                raise RuntimeError(f"Processor initialization failed from '{processor_load_path}'. Error: {str(e_proc)}") from e_proc
        
        logger.info("Florence-2 model and processor loaded successfully.")


    def get_task_prompt_for_quality(self, quality: str = "standard") -> str:
        """
        Get the appropriate task prompt for Florence-2 based on the quality setting.
        
        Args:
            quality: The quality level ("standard", "detailed", or "creative")
            
        Returns:
            str: The appropriate task prompt
        """
        # Florence-2 requires task tokens to be the only tokens in the text input
        # We'll use different task tokens for different quality modes
        if quality == "detailed":
            # <DETAILED_CAPTION> is a special token recognized by Florence-2
            # If that doesn't work, we'll fall back to using <CAPTION> with appropriate generation params
            return "<CAPTION>"
        elif quality == "creative":
            # <image> might not be recognized as a specific task token
            # Use <CAPTION> which is the established task token
            return "<CAPTION>"
        else:  # standard
            # Use standard caption task
            return "<CAPTION>"
    
    def get_generation_params_for_quality(self, quality: str = "standard") -> dict:
        """
        Get the appropriate generation parameters for Florence-2 based on the quality setting.
        
        Args:
            quality: The quality level ("standard", "detailed", or "creative")
            
        Returns:
            dict: Generation parameters
        """
        if quality == "detailed":
            return {
                "max_new_tokens": 300,   # Longer for detailed descriptions
                "num_beams": 5,          # More beams for better quality
                "do_sample": False,      # Deterministic for factual details
                "length_penalty": 1.5,   # Encourage longer generations
                "repetition_penalty": 1.2 # Avoid repetition
            }
        elif quality == "creative":
            return {
                "max_new_tokens": 200,   # Medium length for creative descriptions
                "num_beams": 3,          # Fewer beams allows more variation
                "do_sample": True,       # Enable sampling for creativity
                "temperature": 0.9,      # Higher temperature for more creativity
                "top_p": 0.95,           # Higher top_p for more variety
                "top_k": 50,             # Add top_k sampling for diversity
                "repetition_penalty": 1.1 # Light repetition penalty
            }
        else:  # standard
            return {
                "max_new_tokens": 100,   # Shorter for concise descriptions
                "num_beams": 3,          # Standard beam search
                "do_sample": False,      # Deterministic for consistency
                "length_penalty": 1.0    # Neutral length penalty
            }
    
    def analyze_image(self, image_path: str, quality: str = "standard") -> Tuple[str, Optional[str]]:
        """
        Analyze an image using the Florence-2 model with quality-specific settings.
        
        Args:
            image_path: Path to the image file
            quality: Quality level - "standard", "detailed", or "creative"
            
        Returns:
            Tuple[str, Optional[str]]: (description, clean_caption)
        """
        logger.info(f"Florence2Model using quality mode: '{quality}'")
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

            # Generate caption based on quality setting
            try:
                # Get quality-specific task prompt and generation parameters
                task_prompt = self.get_task_prompt_for_quality(quality)
                generation_params = self.get_generation_params_for_quality(quality)
                
                # Caption generation
                caption_inputs = self.processor(
                    text=task_prompt,
                    images=image,
                    return_tensors="pt"
                ).to(self.device, self.torch_dtype)

                with torch.inference_mode():
                    caption_ids = self.model.generate(
                        input_ids=caption_inputs["input_ids"],
                        pixel_values=caption_inputs["pixel_values"],
                        **generation_params
                    )
                
                caption = self.processor.batch_decode(
                    caption_ids,
                    skip_special_tokens=True
                )[0]

                # Object detection (for all quality levels)
                od_inputs = self.processor(
                    text="<OD>",  # Standard task prompt for object detection
                    images=image,
                    return_tensors="pt"
                ).to(self.device, self.torch_dtype)

                with torch.inference_mode():
                    od_ids = self.model.generate(
                        input_ids=od_inputs["input_ids"],
                        pixel_values=od_inputs["pixel_values"],
                        max_new_tokens=128,  # Max tokens for OD output
                        num_beams=3,
                        do_sample=False  # OD is usually deterministic
                    )
                
                od_text = self.processor.batch_decode(od_ids, skip_special_tokens=False)[0]
                
                # Post-process for object detection
                objects_data = self.processor.post_process_generation(
                    od_text,
                    task="<OD>",
                    image_size=(image.width, image.height)
                )
                
                # Extract objects
                objects = []
                if isinstance(objects_data, dict) and '<OD>' in objects_data:
                    od_results = objects_data['<OD>']
                    if 'labels' in od_results and 'bboxes' in od_results:
                        for label, bbox in zip(od_results['labels'], od_results['bboxes']):
                            objects.append(f"{label} at {bbox}")

            except Exception as e:
                logger.error(f"Error generating analysis: {str(e)}")
                return "Error: Failed to generate image analysis.", None

            # Process and return results
            try:
                caption = self.clean_output(caption)
                objects_str = ", ".join(objects) if objects else "No distinct objects detected."
                
                # Format the description based on quality
                if quality == "detailed":
                    # Apply post-processing to make the caption more detailed if Florence-2 didn't do it
                    # We can expand the caption with additional details and formatting
                    detailed_caption = caption
                    # Add more descriptive elements if the caption is too short for detailed mode
                    if len(detailed_caption.split()) < 30 and objects:
                        detailed_caption += f". The image contains {len(objects)} identifiable objects including {objects_str[:100]}..."
                    
                    description = f"Detailed Description: {detailed_caption}"
                    if objects:
                        description += f"\n\nDetected Objects: {objects_str}"
                        description += f"\n\nObject Count: {len(objects)}"
                    
                    # Add additional analysis for detailed mode
                    try:
                        width, height = image.size
                        description += f"\n\nImage Dimensions: {width}x{height} pixels"
                        description += f"\n\nImage Aspect Ratio: {width/height:.2f}"
                    except:
                        pass
                    
                elif quality == "creative":
                    # Apply post-processing to make the caption more creative
                    creative_framing = [
                        "This captivating image reveals",
                        "An artistic perspective showing",
                        "A visual journey featuring",
                        "This evocative scene presents",
                        "A compelling visual narrative of"
                    ]
                    import random
                    framing = random.choice(creative_framing)
                    
                    # Replace the beginning of the caption with a more creative framing
                    if len(caption) > 10:
                        words = caption.split()
                        # Remove generic starts like "This is" or "The image shows"
                        if len(words) > 3 and words[0].lower() in ["this", "the", "an", "a"]:
                            # Skip common plain beginnings
                            if words[1].lower() in ["is", "shows", "depicts", "contains", "image"]:
                                # Start from 2 or 3 words in
                                if len(words) > 4 and words[2].lower() in ["of", "a", "an", "the", "showing"]:
                                    creative_caption = f"{framing} {' '.join(words[3:])}"
                                else:
                                    creative_caption = f"{framing} {' '.join(words[2:])}"
                            else:
                                creative_caption = f"{framing} {' '.join(words[1:])}"
                        else:
                            creative_caption = f"{framing} {caption}"
                    else:
                        creative_caption = caption
                        
                    description = f"Creative Interpretation: {creative_caption}"
                    if objects:
                        description += f"\n\nFeatured Elements: {objects_str}"
                else:  # standard
                    # Keep standard caption simple and concise
                    # If caption is too long for standard mode, truncate it
                    if len(caption.split()) > 50:
                        short_caption = " ".join(caption.split()[:50]) + "..."
                    else:
                        short_caption = caption
                        
                    description = f"Description: {short_caption}"
                    if objects:
                        description += f"\n\nDetected objects: {objects_str}"
                
                description += f"\n\nGenerated using Florence-2 ({quality} mode)"
                
                return description, caption
            except Exception as e:
                logger.error(f"Error processing model output: {str(e)}")
                return "Error: Failed to process model output.", None

        except Exception as e:
            logger.error(f"Error analyzing image with Florence-2: {str(e)}")
            return "Error: An unexpected error occurred.", None

    def analyze_images_batch(self, image_paths: List[str], quality: str = "standard") -> List[Tuple[str, Optional[str]]]:
        """Analyze a batch of images using the Florence-2 model by processing them individually."""
        if not image_paths:
            return []
        
        results = []
        for image_path in image_paths:
            # Call the existing single-image analysis method
            description, clean_caption = self.analyze_image(image_path, quality=quality)
            results.append((description, clean_caption))
        return results

    @classmethod
    def is_available(cls) -> bool:
        """Check if the model can be initialized with current environment."""
        try:
            # Check for required dependencies except timm which might have issues
            import torch
            import transformers
            
            # Test a direct import from transformers for the necessary classes
            try:
                from transformers import AutoProcessor, AutoModelForCausalLM
                has_transformers = True
            except ImportError:
                logger.error("Required transformers classes (AutoProcessor, AutoModelForCausalLM) not found. Please update transformers.")
                has_transformers = False
                
            # Check CUDA availability
            has_cuda = torch.cuda.is_available()
            if not has_cuda:
                logger.warning("CUDA not available. Model will run in CPU mode with greatly reduced performance.")
                
            return has_transformers
        except Exception as e:
            logger.error(f"Failed to check model availability: {str(e)}")
            return False
