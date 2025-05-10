import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer # <--- Add AutoTokenizer
from qwen_vl_utils import process_vision_info
from PIL import Image
import requests
from io import BytesIO
import logging
import os # <--- Add this import

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Determine torch_dtype
        if device.type == 'cuda' and hasattr(torch, 'bfloat16'):
            torch_dtype = torch.bfloat16
            logger.info("Using torch.bfloat16 for model.")
        elif device.type == 'cuda' and hasattr(torch, 'float16'):
            torch_dtype = torch.float16
            logger.info("Using torch.float16 for model (bfloat16 not available).")
        else:
            torch_dtype = torch.float32
            logger.info("Using torch.float32 for model (CUDA bfloat16/float16 not available).")

        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        logger.info(f"Loading model: {model_id}")

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto", # Let transformers handle device placement
            "trust_remote_code": True # Qwen models often require this
        }

        # Enable Flash Attention 2 if using bfloat16/float16 on CUDA and flash_attn is available
        if device.type == 'cuda' and (torch_dtype == torch.bfloat16 or torch_dtype == torch.float16):
            try:
                import flash_attn 
                logger.info("Attempting to enable Flash Attention 2...")
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.warning("flash_attn library not found. Flash Attention 2 cannot be enabled. Install with 'pip install flash-attn --no-build-isolation'")
            except Exception as e:
                logger.warning(f"Could not enable Flash Attention 2: {e}. Proceeding without it.")
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs
        )
        logger.info(f"Model loaded successfully with kwargs: {model_kwargs}")

        logger.info(f"Loading processor for: {model_id}")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        logger.info("Processor loaded successfully.")

        # Use a local image
        image_path = "data/review/Shadow_Assassin_00159_.png" 
        logger.info(f"Loading local image from: {image_path}")
        if not os.path.exists(image_path):
            logger.error(f"Local image not found at {image_path}. Please ensure the path is correct.")
            return
        
        pil_image = Image.open(image_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        logger.info("Local image loaded and processed into PIL format.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image}, 
                    {"type": "text", "text": "Describe this image in detail."}, # Changed prompt for better testing
                ],
            }
        ]
        logger.info(f"Messages prepared: {messages}")

        logger.info("Preparing inputs for inference...")
        # Ensure tokenizer is loaded for chat template
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        text_template = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        logger.info(f"Text template for processor: {text_template}")
        
        image_inputs_processed, video_inputs_processed = process_vision_info(messages) # Assuming process_vision_info is globally available from qwen_vl_utils
        logger.info(f"Vision info processed. Image inputs: {type(image_inputs_processed)}, Video inputs: {type(video_inputs_processed)}")

        inputs = processor(
            text=[text_template],
            images=image_inputs_processed,
            videos=video_inputs_processed, # Should be None for this example
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to the model's device (especially if device_map="auto" was used for model)
        # If model is on multiple devices, inputs should go to the device of the first parameter.
        # For simplicity with device_map="auto", we assume inputs should go to the primary CUDA device if available.
        inputs = inputs.to(device)
        logger.info(f"Inputs prepared and moved to device: {inputs.input_ids.device}")


        logger.info("Starting inference...")
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        logger.info("Inference complete.")
        
        # Trim input tokens from generated_ids
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        logger.info(f"Generated output: {output_text}")
        print(f"Output: {output_text[0]}")

    except ImportError as e:
        logger.error(f"An import error occurred: {e}. Please ensure all dependencies are installed correctly.")
        logger.error("Key dependencies: transformers (latest from git), accelerate, qwen-vl-utils[decord]==0.0.8, torch, Pillow.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
