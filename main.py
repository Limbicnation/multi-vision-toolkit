import argparse
import json
import os
import logging
import threading
import queue
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import shutil
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Union, Any, Callable
from dataclasses import dataclass, field
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ImageAnalysisResult:
    """Data class to store image analysis results"""
    description: str
    clean_caption: Optional[str] = None
    objects: List[Dict] = field(default_factory=list)
    regions: List[Dict] = field(default_factory=list)
    ocr_text: Optional[str] = None
    grounding_results: Dict = field(default_factory=dict)
    visualizations: Dict[str, Image.Image] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary for JSON serialization"""
        result = {
            "description": self.description,
            "clean_caption": self.clean_caption,
            "objects": self.objects,
            "regions": self.regions,
            "ocr_text": self.ocr_text,
            "grounding_results": self.grounding_results
        }
        # Visualizations are PIL images, not serializable
        return result
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'ImageAnalysisResult':
        """Create result from dictionary"""
        return cls(
            description=data.get("description", ""),
            clean_caption=data.get("clean_caption"),
            objects=data.get("objects", []),
            regions=data.get("regions", []),
            ocr_text=data.get("ocr_text"),
            grounding_results=data.get("grounding_results", {})
        )

class BaseVisionModel:
    """Base class for vision models"""
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def analyze_image(self, image_path: str, tasks: List[str] = None) -> ImageAnalysisResult:
        """
        Analyze image with specified tasks
        
        Args:
            image_path: Path to image file
            tasks: List of tasks to perform (caption, objects, regions, ocr, grounding)
                  If None, performs basic captioning
        
        Returns:
            ImageAnalysisResult object containing results
        """
        raise NotImplementedError("Subclasses must implement analyze_image")
    
    def batch_process(self, 
                      image_paths: List[str], 
                      tasks: List[str] = None,
                      max_workers: int = 4,
                      callback: Callable[[str, ImageAnalysisResult], None] = None) -> Dict[str, ImageAnalysisResult]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of paths to images
            tasks: List of tasks to perform on each image
            max_workers: Maximum number of parallel workers
            callback: Function to call after each image is processed
            
        Returns:
            Dictionary mapping image paths to results
        """
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.analyze_image, path, tasks): path 
                for path in image_paths
            }
            
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results[path] = result
                    if callback:
                        callback(path, result)
                except Exception as e:
                    logger.error(f"Error processing {path}: {str(e)}")
                    if callback:
                        callback(path, None)
        
        return results

class Florence2Model(BaseVisionModel):
    """Florence-2 vision model implementation"""
    def __init__(self, model_size: str = "base", device: str = None):
        super().__init__(device)
        try:
            # Import here to avoid dependency if not using this model
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            logger.info(f"Initializing Florence2 model (size: {model_size})")
            
            # Map model size to actual model name
            model_map = {
                "base": "microsoft/florence-2-base",
                "large": "microsoft/florence-2-large",
                "instruct": "microsoft/florence-2-instruct"
            }
            model_name = model_map.get(model_size.lower(), model_map["base"])
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(model_name).to(self.device)
            logger.info(f"Florence2 model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Florence2 model: {str(e)}")
            raise

    def analyze_image(self, image_path: str, tasks: List[str] = None) -> ImageAnalysisResult:
        """Analyze image with Florence2 model"""
        if tasks is None:
            tasks = ["caption"]
            
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            result = ImageAnalysisResult(description="")
            
            # Perform requested tasks
            if "caption" in tasks:
                result.description, result.clean_caption = self._generate_caption(image)
                
            if "objects" in tasks:
                result.objects = self._detect_objects(image)
                if result.objects:
                    result.visualizations["objects"] = self._visualize_objects(image, result.objects)
                
            if "regions" in tasks:
                result.regions = self._generate_region_captions(image)
                if result.regions:
                    result.visualizations["regions"] = self._visualize_regions(image, result.regions)
                
            if "ocr" in tasks:
                result.ocr_text = self._perform_ocr(image)
                
            if "grounding" in tasks and "phrase" in tasks:
                phrases = ["person", "car", "dog", "cat", "building"]  # Default phrases to ground
                result.grounding_results = self._ground_phrases(image, phrases)
                if result.grounding_results:
                    result.visualizations["grounding"] = self._visualize_grounding(image, result.grounding_results)
                
            return result
            
        except Exception as e:
            logger.error(f"Error in Florence2 analysis: {str(e)}")
            raise

    def _generate_caption(self, image: Image.Image) -> Tuple[str, str]:
        """Generate caption for image"""
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate basic caption
            caption_prompt = "Generate a detailed caption for this image."
            prompt_ids = self.processor.tokenizer(caption_prompt, return_tensors="pt").to(self.device)
            
            output_ids = self.model.generate(
                **inputs,
                decoder_input_ids=prompt_ids.input_ids,
                max_length=100,
                num_beams=5
            )
            
            description = self.processor.decode(output_ids[0], skip_special_tokens=True)
            
            # Generate clean caption (suitable for training)
            clean_prompt = "Generate a concise caption for this image."
            clean_prompt_ids = self.processor.tokenizer(clean_prompt, return_tensors="pt").to(self.device)
            
            clean_output_ids = self.model.generate(
                **inputs,
                decoder_input_ids=clean_prompt_ids.input_ids,
                max_length=50,
                num_beams=3
            )
            
            clean_caption = self.processor.decode(clean_output_ids[0], skip_special_tokens=True)
            
            return description, clean_caption
            
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return "Error generating caption", None

    def _detect_objects(self, image: Image.Image) -> List[Dict]:
        """Detect objects in image"""
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate objects
            object_prompt = "List all objects in this image with their bounding boxes."
            prompt_ids = self.processor.tokenizer(object_prompt, return_tensors="pt").to(self.device)
            
            output_ids = self.model.generate(
                **inputs,
                decoder_input_ids=prompt_ids.input_ids,
                max_length=200,
                num_beams=5
            )
            
            object_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
            
            # Parse object text into structured format (simplified example)
            # In a real implementation, you might need more sophisticated parsing logic
            objects = []
            try:
                # Simple parsing assuming format: "object_name: [x1, y1, x2, y2]"
                for line in object_text.split('\n'):
                    if ':' in line and '[' in line and ']' in line:
                        name, coords = line.split(':', 1)
                        coords = coords.strip()
                        if coords.startswith('[') and coords.endswith(']'):
                            # Extract the coordinates
                            coords = coords[1:-1].split(',')
                            if len(coords) == 4:
                                x1, y1, x2, y2 = [float(c.strip()) for c in coords]
                                objects.append({
                                    "name": name.strip(),
                                    "bbox": [x1, y1, x2, y2]
                                })
            except Exception as e:
                logger.warning(f"Error parsing object detection output: {str(e)}")
                
            return objects
            
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            return []

    def _generate_region_captions(self, image: Image.Image) -> List[Dict]:
        """Generate captions for image regions"""
        # Simplified implementation
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate region captions
            region_prompt = "Describe different regions of this image with their bounding boxes."
            prompt_ids = self.processor.tokenizer(region_prompt, return_tensors="pt").to(self.device)
            
            output_ids = self.model.generate(
                **inputs,
                decoder_input_ids=prompt_ids.input_ids,
                max_length=300,
                num_beams=5
            )
            
            region_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
            
            # Parse region text into structured format
            regions = []
            # Similar parsing logic to objects, but with region descriptions
            
            return regions
            
        except Exception as e:
            logger.error(f"Error generating region captions: {str(e)}")
            return []

    def _perform_ocr(self, image: Image.Image) -> Optional[str]:
        """Perform OCR on image"""
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate OCR text
            ocr_prompt = "Extract all text from this image."
            prompt_ids = self.processor.tokenizer(ocr_prompt, return_tensors="pt").to(self.device)
            
            output_ids = self.model.generate(
                **inputs,
                decoder_input_ids=prompt_ids.input_ids,
                max_length=200,
                num_beams=5
            )
            
            ocr_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
            return ocr_text
            
        except Exception as e:
            logger.error(f"Error performing OCR: {str(e)}")
            return None

    def _ground_phrases(self, image: Image.Image, phrases: List[str]) -> Dict:
        """Ground phrases in image"""
        # Simplified implementation
        return {}

    def _visualize_objects(self, image: Image.Image, objects: List[Dict]) -> Image.Image:
        """Visualize detected objects on image"""
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, obj in enumerate(objects):
            color = colors[i % len(colors)]
            bbox = obj.get("bbox", [0, 0, 0, 0])
            
            # Convert normalized coordinates to pixel coordinates if needed
            img_width, img_height = image.size
            if all(0 <= coord <= 1 for coord in bbox):
                bbox = [
                    bbox[0] * img_width,
                    bbox[1] * img_height,
                    bbox[2] * img_width,
                    bbox[3] * img_height
                ]
            
            draw.rectangle(bbox, outline=color, width=3)
            draw.text((bbox[0], bbox[1] - 15), obj.get("name", ""), fill=color, font=font)
        
        return img_draw

    def _visualize_regions(self, image: Image.Image, regions: List[Dict]) -> Image.Image:
        """Visualize region captions on image"""
        # Similar to visualize_objects but for regions
        return image.copy()

    def _visualize_grounding(self, image: Image.Image, grounding_results: Dict) -> Image.Image:
        """Visualize phrase grounding on image"""
        # Similar to visualize_objects but for grounded phrases
        return image.copy()


class JanusModel(BaseVisionModel):
    """Janus multimodal model implementation"""
    def __init__(self, device: str = None):
        super().__init__(device)
        try:
            # Import here to avoid dependency if not using this model
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            logger.info("Initializing Janus model")
            model_name = "nvidia/janus-3.1-large"  # Example model name
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(model_name).to(self.device)
            logger.info("Janus model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Janus model: {str(e)}")
            raise

    def analyze_image(self, image_path: str, tasks: List[str] = None) -> ImageAnalysisResult:
        """Analyze image with Janus model"""
        if tasks is None:
            tasks = ["caption"]
            
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Basic implementation - in a real scenario, add support for all tasks
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            prompt = "Describe this image in detail."
            prompt_ids = self.processor.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            output_ids = self.model.generate(
                **inputs,
                decoder_input_ids=prompt_ids.input_ids,
                max_length=100,
                num_beams=5
            )
            
            description = self.processor.decode(output_ids[0], skip_special_tokens=True)
            
            # Generate clean caption
            clean_prompt = "Describe this image concisely."
            clean_prompt_ids = self.processor.tokenizer(clean_prompt, return_tensors="pt").to(self.device)
            
            clean_output_ids = self.model.generate(
                **inputs,
                decoder_input_ids=clean_prompt_ids.input_ids,
                max_length=50,
                num_beams=3
            )
            
            clean_caption = self.processor.decode(clean_output_ids[0], skip_special_tokens=True)
            
            result = ImageAnalysisResult(
                description=description,
                clean_caption=clean_caption
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Janus analysis: {str(e)}")
            raise


class ModelManager:
    """Manages model initialization and switching"""
    def __init__(self):
        self.models: Dict[str, BaseVisionModel] = {}
        self._current_model = None
        self._current_model_name = None
        self._current_model_config = None
        
    def initialize_model(self, model_name: str, model_config: Dict = None) -> BaseVisionModel:
        """Initialize a model with error handling and caching"""
        try:
            # Create a key that includes model name and config
            config_key = json.dumps(model_config) if model_config else ""
            cache_key = f"{model_name}_{config_key}"
            
            if cache_key in self.models:
                logger.info(f"Using cached model: {model_name}")
                return self.models[cache_key]
            
            logger.info(f"Initializing new model: {model_name}")
            model_config = model_config or {}
            
            if model_name.lower() == "florence2":
                model_size = model_config.get("size", "base")
                model = Florence2Model(model_size=model_size)
            elif model_name.lower() == "janus":
                try:
                    model = JanusModel()
                except Exception as e:
                    logger.error(f"Failed to load Janus model: {str(e)}")
                    messagebox.showerror("Model Error", 
                        "Failed to load Janus model. Please update transformers:\n"
                        "pip install --upgrade transformers\n"
                        "or install from source:\n"
                        "pip install git+https://github.com/huggingface/transformers.git")
                    raise
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            self.models[cache_key] = model
            return model
            
        except Exception as e:
            logger.error(f"Error initializing model {model_name}: {str(e)}")
            raise

    def get_model(self, model_name: str, model_config: Dict = None) -> BaseVisionModel:
        """Get a model, initializing if necessary"""
        config_key = json.dumps(model_config) if model_config else ""
        cache_key = f"{model_name}_{config_key}"
        
        if self._current_model_name != model_name or self._current_model_config != model_config:
            self._current_model = self.initialize_model(model_name, model_config)
            self._current_model_name = model_name
            self._current_model_config = model_config
        return self._current_model


class DatasetPreparator:
    """Handles dataset preparation and file operations"""
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        
    def is_supported_image(self, filename: str) -> bool:
        """Check if file is a supported image format"""
        return Path(filename).suffix.lower() in self.supported_formats
        
    def create_caption_file(self, image_path: str, caption: str) -> str:
        """Create a caption file for an image"""
        try:
            txt_path = Path(image_path).with_suffix('.txt')
            txt_path.write_text(caption, encoding='utf-8')
            return str(txt_path)
        except Exception as e:
            logger.error(f"Error creating caption file: {str(e)}")
            raise
            
    def create_metadata_file(self, image_path: str, result: ImageAnalysisResult) -> str:
        """Create a JSON metadata file with analysis results"""
        try:
            json_path = Path(image_path).with_suffix('.json')
            data = result.to_dict()
            json_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
            return str(json_path)
        except Exception as e:
            logger.error(f"Error creating metadata file: {str(e)}")
            raise
            
    def batch_process_directory(self, 
                               input_dir: str, 
                               output_dir: str, 
                               model: BaseVisionModel,
                               tasks: List[str] = None,
                               recursive: bool = False,
                               max_workers: int = 4,
                               callback: Callable[[str, ImageAnalysisResult, int, int], None] = None) -> Dict[str, str]:
        """
        Process all images in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            model: Vision model to use
            tasks: List of tasks to perform
            recursive: Whether to recursively process subdirectories
            max_workers: Maximum number of parallel workers
            callback: Function to call after each image is processed
            
        Returns:
            Dictionary mapping input paths to output paths
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Gather all image files
        image_paths = []
        glob_pattern = "**/*" if recursive else "*"
        
        for file_path in input_dir.glob(glob_pattern):
            if file_path.is_file() and self.is_supported_image(file_path):
                image_paths.append(str(file_path))
        
        total_images = len(image_paths)
        logger.info(f"Found {total_images} images to process")
        
        if total_images == 0:
            return {}
            
        # Process images in batch with progress tracking
        results = {}
        processed = 0
        
        def process_callback(path, result):
            nonlocal processed
            processed += 1
            if callback:
                callback(path, result, processed, total_images)
        
        batch_results = model.batch_process(
            image_paths, 
            tasks=tasks,
            max_workers=max_workers,
            callback=process_callback
        )
        
        # Create output files
        for input_path, result in batch_results.items():
            if result is None:
                continue
                
            input_path = Path(input_path)
            rel_path = input_path.relative_to(input_dir)
            output_path = output_dir / rel_path
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy original image
            shutil.copy2(input_path, output_path)
            
            # Create caption file
            if result.clean_caption:
                self.create_caption_file(str(output_path), result.clean_caption)
                
            # Create metadata file
            self.create_metadata_file(str(output_path), result)
            
            # Save visualizations
            for viz_name, viz_image in result.visualizations.items():
                viz_path = output_path.with_suffix(f'.{viz_name}.jpg')
                viz_image.save(viz_path)
                
            results[str(input_path)] = str(output_path)
        
        return results


class BatchProcessingDialog(tk.Toplevel):
    """Dialog for batch processing settings"""
    def __init__(self, parent, available_tasks, model_names, on_start):
        super().__init__(parent)
        self.title("Batch Processing")
        self.geometry("500x500")
        self.resizable(True, True)
        
        self.available_tasks = available_tasks
        self.model_names = model_names
        self.on_start = on_start
        
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.selected_model = tk.StringVar(value=model_names[0] if model_names else "")
        self.selected_tasks = {}
        for task in available_tasks:
            self.selected_tasks[task] = tk.BooleanVar(value=task == "caption")
        
        self.recursive = tk.BooleanVar(value=False)
        self.max_workers = tk.IntVar(value=4)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up dialog UI"""
        frame = ttk.Frame(self, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Input directory
        ttk.Label(frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.input_dir, width=50).grid(row=0, column=1, sticky=tk.EW, pady=5)
        ttk.Button(frame, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)
        
        # Output directory
        ttk.Label(frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=self.output_dir, width=50).grid(row=1, column=1, sticky=tk.EW, pady=5)
        ttk.Button(frame, text="Browse...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)
        
        # Model selection
        ttk.Label(frame, text="Model:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(
            frame, 
            textvariable=self.selected_model,
            values=self.model_names,
            state="readonly"
        ).grid(row=2, column=1, sticky=tk.EW, pady=5)
        
        # Tasks selection
        ttk.Label(frame, text="Tasks:").grid(row=3, column=0, sticky=tk.NW, pady=5)
        
        tasks_frame = ttk.Frame(frame)
        tasks_frame.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        for i, task in enumerate(self.available_tasks):
            ttk.Checkbutton(
                tasks_frame,
                text=task.capitalize(),
                variable=self.selected_tasks[task]
            ).grid(row=i // 2, column=i % 2, sticky=tk.W, padx=5, pady=2)
        
        # Options
        options_frame = ttk.LabelFrame(frame, text="Options")
        options_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=10, padx=5)
        
        ttk.Checkbutton(
            options_frame,
            text="Process subdirectories recursively",
            variable=self.recursive
        ).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(options_frame, text="Max parallel workers:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(
            options_frame,
            from_=1,
            to=16,
            textvariable=self.max_workers,
            width=5
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Start Processing", command=self.start_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)
        
    def browse_input(self):
        """Browse for input directory"""
        directory = filedialog.askdirectory()
        if directory:
            self.input_dir.set(directory)
            
    def browse_output(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)
            
    def start_processing(self):
        """Start batch processing"""
        input_dir = self.input_dir.get()
        output_dir = self.output_dir.get()
        
        if not input_dir or not output_dir:
            messagebox.showerror("Error", "Please select input and output directories")
            return
            
        tasks = [task for task, var in self.selected_tasks.items() if var.get()]
        
        if not tasks:
            messagebox.showerror("Error", "Please select at least one task")
            return
        
        self.destroy()
        
        # Call the start function
        self.on_start(
            input_dir=input_dir,
            output_dir=output_dir,
            model_name=self.selected_model.get(),
            tasks=tasks,
            recursive=self.recursive.get(),
            max_workers=self.max_workers.get()
        )


class ProgressDialog(tk.Toplevel):
    """Dialog for showing batch processing progress"""
    def __init__(self, parent, title="Processing", max_value=100):
        super().__init__(parent)
        self.title(title)
        self.geometry("400x150")
        self.resizable(False, False)
        
        self.setup_ui(max_value)
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
    def setup_ui(self, max_value):
        """Set up progress dialog UI"""
        frame = ttk.Frame(self, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_var = tk.StringVar(value="Initializing...")
        ttk.Label(frame, textvariable=self.status_var).pack(pady=(0, 10))
        
        self.progress = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=350, mode='determinate', maximum=max_value)
        self.progress.pack(pady=10)
        
        self.detail_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.detail_var).pack(pady=(10, 0))
        
        self.cancel_button = ttk.Button(frame, text="Cancel", command=self.on_cancel)
        self.cancel_button.pack(pady=10)
        
        self.cancelled = False
        
    def update_progress(self, value, status=None, detail=None):
        """Update progress bar and status"""
        self.progress['value'] = value
        
        if status:
            self.status_var.set(status)
            
        if detail:
            self.detail_var.set(detail)
            
        self.update()
        
    def on_cancel(self):
        """Handle cancel button click"""
        if messagebox.askyesno("Cancel", "Are you sure you want to cancel the operation?"):
            self.cancelled = True
            self.status_var.set("Cancelling...")
            self.cancel_button['state'] = 'disabled'


class ReviewGUI:
    """Main GUI application for reviewing images"""
    def __init__(
        self, 
        review_dir: str, 
        approved_dir: str, 
        rejected_dir: str, 
        trigger_word: Optional[str] = None,
        model_name: str = "florence2",
        model_config: Dict = None
    ):
        logger.info(f"Initializing ReviewGUI with model: {model_name}")
        self.review_dir = Path(review_dir)
        self.approved_dir = Path(approved_dir)
        self.rejected_dir = Path(rejected_dir)
        self.trigger_word = trigger_word
        
        self.model_name = model_name
        self.model_config = model_config or {"size": "base"}
        self.model_manager = ModelManager()
        
        try:
            self.model = self.model_manager.get_model(model_name, self.model_config)
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize model {model_name}. Falling back to Florence2.")
            self.model_name = "florence2"
            self.model_config = {"size": "base"}
            self.model = self.model_manager.get_model("florence2", self.model_config)
        
        self.dataset_prep = DatasetPreparator()
        
        # Analysis task settings
        self.tasks = ["caption"]
        
        # Create directories
        for dir_path in [self.review_dir, self.approved_dir, self.rejected_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created/verified directory: {dir_path}")
        
        self.root = tk.Tk()
        self.root.title("AI Training Dataset Preparation Tool")
        self.root.geometry("1200x800")
        
        # Set up task processing queue for background tasks
        self.task_queue = queue.Queue()
        self.stop_worker = threading.Event()
        self.worker_thread = threading.Thread(target=self._background_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        self.setup_gui()
        self.load_items()

    def setup_gui(self):
        """Setup GUI components with error handling"""
        try:
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            
            # Main frame
            main_frame = ttk.Frame(self.root)
            main_frame.grid(row=0, column=0, sticky="nsew")
            main_frame.columnconfigure(0, weight=1)
            main_frame.rowconfigure(1, weight=1)
            
            # Toolbar frame
            toolbar_frame = ttk.Frame(main_frame)
            toolbar_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
            
            # Model selection
            model_frame = ttk.Frame(toolbar_frame)
            model_frame.pack(side=tk.LEFT, padx=5)
            
            ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=5)
            self.model_var = tk.StringVar(value=self.model_name)
            model_combo = ttk.Combobox(
                model_frame, 
                textvariable=self.model_var,
                values=["florence2", "janus"],
                state="readonly",
                width=10
            )
            model_combo.pack(side=tk.LEFT, padx=5)
            model_combo.bind('<<ComboboxSelected>>', self._on_model_change)
            
            # Model size (for Florence2)
            self.size_frame = ttk.Frame(toolbar_frame)
            self.size_frame.pack(side=tk.LEFT, padx=5)
            
            ttk.Label(self.size_frame, text="Size:").pack(side=tk.LEFT, padx=5)
            self.size_var = tk.StringVar(value=self.model_config.get("size", "base"))
            size_combo = ttk.Combobox(
                self.size_frame, 
                textvariable=self.size_var,
                values=["base", "large", "instruct"],
                state="readonly",
                width=8
            )
            size_combo.pack(side=tk.LEFT, padx=5)
            size_combo.bind('<<ComboboxSelected>>', self._on_size_change)
            
            # Task selection
            task_frame = ttk.Frame(toolbar_frame)
            task_frame.pack(side=tk.LEFT, padx=10)
            
            ttk.Label(task_frame, text="Tasks:").pack(side=tk.LEFT, padx=5)
            self.task_vars = {}
            for task in ["caption", "objects", "regions", "ocr", "grounding"]:
                var = tk.BooleanVar(value=task in self.tasks)
                cb = ttk.Checkbutton(
                    task_frame, 
                    text=task.capitalize(),
                    variable=var,
                    command=self._on_task_change
                )
                cb.pack(side=tk.LEFT, padx=5)
                self.task_vars[task] = var
            
            # Batch processing button
            ttk.Button(
                toolbar_frame, 
                text="Batch Process...",
                command=self._show_batch_dialog
            ).pack(side=tk.RIGHT, padx=5)
            
            # Content frame with scrollbar
            content_frame = ttk.Frame(main_frame)
            content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
            content_frame.columnconfigure(0, weight=1)
            content_frame.rowconfigure(0, weight=1)
            
            # Canvas for scrolling
            self.canvas = tk.Canvas(content_frame)
            scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=scrollbar.set)
            
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Frame inside canvas for content
            self.scroll_frame = ttk.Frame(self.canvas)
            self.canvas_frame = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
            
            self.canvas.bind("<Configure>", self._on_canvas_configure)
            self.scroll_frame.bind("<Configure>", self._on_frame_configure)
            
            # Mouse wheel scrolling
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
            
            # Image and caption display
            self.img_label = ttk.Label(self.scroll_frame)
            self.img_label.pack(pady=10)
            
            self.caption_frame = ttk.LabelFrame(self.scroll_frame, text="Analysis Results")
            self.caption_frame.pack(fill=tk.X, padx=10, pady=5, expand=True)
            
            self.caption = tk.StringVar()
            caption_label = ttk.Label(
                self.caption_frame, 
                textvariable=self.caption, 
                wraplength=800,
                justify=tk.LEFT
            )
            caption_label.pack(padx=10, pady=10, fill=tk.X)
            
            # Task result frames
            self.task_frames = {}
            
            # Control buttons
            btn_frame = ttk.Frame(main_frame)
            btn_frame.grid(row=2, column=0, pady=10)
            
            approve_btn = ttk.Button(btn_frame, text="Approve (A)", command=self.approve)
            approve_btn.pack(side=tk.LEFT, padx=5)
            
            reject_btn = ttk.Button(btn_frame, text="Reject (R)", command=self.reject)
            reject_btn.pack(side=tk.LEFT, padx=5)
            
            skip_btn = ttk.Button(btn_frame, text="Skip (S)", command=self.next_item)
            skip_btn.pack(side=tk.LEFT, padx=5)
            
            # Status bar
            self.status_var = tk.StringVar()
            status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
            status_bar.grid(row=1, column=0, sticky="ew")
            
            # Keyboard shortcuts
            self.root.bind('a', lambda e: self.approve())
            self.root.bind('r', lambda e: self.reject())
            self.root.bind('s', lambda e: self.next_item())
            self.root.bind('<Left>', lambda e: self.prev_item())
            self.root.bind('<Right>', lambda e: self.next_item())
            
        except Exception as e:
            logger.error(f"Error setting up GUI: {str(e)}")
            raise

    def _on_canvas_configure(self, event):
        """Update canvas scrollregion when the canvas is resized"""
        self.canvas.itemconfig(self.canvas_frame, width=event.width)
        
    def _on_frame_configure(self, event):
        """Update the canvas scrollregion when the frame size changes"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_model_change(self, event):
        """Handle model switching with error handling"""
        try:
            new_model = self.model_var.get()
            if new_model != self.model_name:
                logger.info(f"Switching model from {self.model_name} to {new_model}")
                
                # Update UI based on model selection
                if new_model == "florence2":
                    self.size_frame.pack(side=tk.LEFT, padx=5)
                else:
                    self.size_frame.pack_forget()
                
                self.model_name = new_model
                self.model_config = {"size": self.size_var.get()} if new_model == "florence2" else {}
                
                try:
                    self.model = self.model_manager.get_model(new_model, self.model_config)
                    if self.items:
                        self.show_current()
                except Exception as e:
                    logger.error(f"Failed to switch to model {new_model}: {str(e)}")
                    messagebox.showerror("Error", f"Failed to switch to {new_model}. Reverting to previous model.")
                    self.model_var.set(self.model_name)
        except Exception as e:
            logger.error(f"Error in model change handler: {str(e)}")
            raise

    def _on_size_change(self, event):
        """Handle model size change"""
        try:
            new_size = self.size_var.get()
            if new_size != self.model_config.get("size"):
                logger.info(f"Changing model size to {new_size}")
                self.model_config["size"] = new_size
                
                try:
                    self.model = self.model_manager.get_model(self.model_name, self.model_config)
                    if self.items:
                        self.show_current()
                except Exception as e:
                    logger.error(f"Failed to change model size: {str(e)}")
                    messagebox.showerror("Error", f"Failed to change model size. Reverting to previous size.")
                    self.size_var.set(self.model_config.get("size", "base"))
        except Exception as e:
            logger.error(f"Error in size change handler: {str(e)}")
            raise

    def _on_task_change(self):
        """Handle task selection change"""
        try:
            self.tasks = [task for task, var in self.task_vars.items() if var.get()]
            logger.info(f"Tasks changed to: {self.tasks}")
            
            if self.items:
                self.show_current()
        except Exception as e:
            logger.error(f"Error in task change handler: {str(e)}")
            raise

    def _show_batch_dialog(self):
        """Show batch processing dialog"""
        try:
            available_tasks = list(self.task_vars.keys())
            model_names = ["florence2", "janus"]
            
            dialog = BatchProcessingDialog(
                self.root,
                available_tasks,
                model_names,
                self._start_batch_processing
            )
            dialog.focus_set()
        except Exception as e:
            logger.error(f"Error showing batch dialog: {str(e)}")
            raise

    def _start_batch_processing(self, input_dir, output_dir, model_name, tasks, recursive, max_workers):
        """Start batch processing with progress dialog"""
        try:
            # Create model for batch processing
            model_config = {"size": self.size_var.get()} if model_name == "florence2" else {}
            model = self.model_manager.get_model(model_name, model_config)
            
            # Find all images
            input_path = Path(input_dir)
            glob_pattern = "**/*" if recursive else "*"
            image_paths = [
                str(p) for p in input_path.glob(glob_pattern)
                if p.is_file() and self.dataset_prep.is_supported_image(p)
            ]
            
            total_images = len(image_paths)
            if total_images == 0:
                messagebox.showinfo("Info", "No supported image files found for processing.")
                return
                
            # Create progress dialog
            progress_dialog = ProgressDialog(
                self.root, 
                title="Batch Processing", 
                max_value=total_images
            )
            
            # Start processing in background thread
            threading.Thread(
                target=self._batch_process_thread,
                args=(
                    progress_dialog,
                    input_dir,
                    output_dir,
                    model,
                    tasks,
                    recursive,
                    max_workers
                ),
                daemon=True
            ).start()
            
        except Exception as e:
            logger.error(f"Error starting batch processing: {str(e)}")
            messagebox.showerror("Error", f"Failed to start batch processing: {str(e)}")

    def _batch_process_thread(self, progress_dialog, input_dir, output_dir, model, tasks, recursive, max_workers):
        """Background thread for batch processing"""
        try:
            def progress_callback(path, result, processed, total):
                if progress_dialog.cancelled:
                    return False
                    
                rel_path = Path(path).name
                progress_dialog.update_progress(
                    processed,
                    f"Processing {processed} of {total}",
                    f"Current: {rel_path}"
                )
                return True
                
            # Start batch processing
            results = self.dataset_prep.batch_process_directory(
                input_dir=input_dir,
                output_dir=output_dir,
                model=model,
                tasks=tasks,
                recursive=recursive,
                max_workers=max_workers,
                callback=progress_callback
            )
            
            # Close progress dialog and show results
            if not progress_dialog.cancelled:
                self.root.after(0, progress_dialog.destroy)
                self.root.after(
                    0, 
                    lambda: messagebox.showinfo(
                        "Batch Processing Complete", 
                        f"Processed {len(results)} images successfully."
                    )
                )
            else:
                self.root.after(0, progress_dialog.destroy)
                
        except Exception as e:
            logger.error(f"Error in batch processing thread: {str(e)}")
            self.root.after(
                0, 
                lambda: messagebox.showerror(
                    "Error", 
                    f"Error during batch processing: {str(e)}"
                )
            )
            self.root.after(0, progress_dialog.destroy)

    def _background_worker(self):
        """Background worker thread for processing tasks"""
        while not self.stop_worker.is_set():
            try:
                task, args, callback = self.task_queue.get(timeout=0.5)
                try:
                    result = task(*args)
                    if callback:
                        self.root.after(0, lambda: callback(result))
                except Exception as e:
                    logger.error(f"Error in background task: {str(e)}")
                    if callback:
                        self.root.after(0, lambda: callback(None, str(e)))
                finally:
                    self.task_queue.task_done()
            except queue.Empty:
                pass

    def load_items(self):
        """Load image items with error handling"""
        self.items = []
        self.current = 0
        
        try:
            if not self.review_dir.exists():
                logger.warning(f"Review directory not found: {self.review_dir}")
                return
                
            for f in self.review_dir.iterdir():
                if self.dataset_prep.is_supported_image(f):
                    img_path = f
                    base_name = f.stem
                    json_path = f.parent / f"{base_name}_for_review.json"
                    
                    if not json_path.exists():
                        json_path.write_text(
                            json.dumps({"results": {"caption": ""}}, indent=2),
                            encoding='utf-8'
                        )
                    
                    self.items.append((base_name, json_path, img_path))
            
            if self.items:
                self.status_var.set(f"Loaded {len(self.items)} images for review")
                self.show_current()
            else:
                logger.info("No supported image files found for review.")
                self.status_var.set("No images found for review")
                messagebox.showinfo("Info", "No images found for review.")
        except Exception as e:
            logger.error(f"Error loading items: {str(e)}")
            raise

    def show_current(self):
        """Display current image with error handling"""
        if not self.items:
            return
            
        try:
            # Clear previous task frames
            for frame in self.task_frames.values():
                frame.destroy()
            self.task_frames = {}
            
            # Get current item
            base_name, json_path, img_path = self.items[self.current]
            
            self.status_var.set(f"Reviewing item {self.current + 1} of {len(self.items)}: {img_path.name}")
            
            # Schedule image analysis in background thread
            self.caption.set("Analyzing image...")
            self.root.title(f"Analyzing... ({self.current + 1}/{len(self.items)})")
            
            self.task_queue.put((
                self.model.analyze_image,
                (str(img_path), self.tasks),
                self._on_analysis_complete
            ))
            
            # Load and display image
            img = Image.open(img_path)
            img.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(img)
            self.img_label.configure(image=photo)
            self.img_label.image = photo
            
        except Exception as e:
            logger.error(f"Error showing current item: {str(e)}")
            self.caption.set(f"Error: {str(e)}")

    def _on_analysis_complete(self, result, error=None):
        """Handle completed image analysis"""
        if not self.items:
            return
            
        try:
            base_name, json_path, img_path = self.items[self.current]
            
            if error:
                self.caption.set(f"Error analyzing image: {error}")
                messagebox.showwarning("Warning", f"Error analyzing image: {error}")
                return
                
            if not result:
                self.caption.set("Failed to analyze image")
                return
                
            # Add trigger word if specified
            if self.trigger_word and result.clean_caption:
                result.clean_caption = f"{self.trigger_word}, {result.clean_caption}"
            
            # Save analysis results
            data = {"results": result.to_dict()}
            json_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
            
            if result.clean_caption:
                self.dataset_prep.create_caption_file(str(img_path), result.clean_caption)
            
            # Display results
            self.caption.set(f"Caption:\n{result.description}")
            self.root.title(f"Review {self.current + 1}/{len(self.items)}")
            
            # Show task-specific results
            self._display_task_results(result)
            
        except Exception as e:
            logger.error(f"Error handling analysis result: {str(e)}")
            self.caption.set(f"Error: {str(e)}")

    def _display_task_results(self, result: ImageAnalysisResult):
        """Display task-specific results"""
        try:
            # Object detection results
            if result.objects and "objects" in self.tasks:
                frame = ttk.LabelFrame(self.scroll_frame, text="Object Detection")
                frame.pack(fill=tk.X, padx=10, pady=5, expand=True)
                self.task_frames["objects"] = frame
                
                # Display object visualization if available
                if "objects" in result.visualizations:
                    img = result.visualizations["objects"]
                    img.thumbnail((800, 600))
                    photo = ImageTk.PhotoImage(img)
                    img_label = ttk.Label(frame, image=photo)
                    img_label.image = photo
                    img_label.pack(pady=5)
                
                # Display object list
                objects_text = "\n".join([f"• {obj['name']}" for obj in result.objects])
                if objects_text:
                    ttk.Label(
                        frame, 
                        text=f"Detected objects:\n{objects_text}",
                        wraplength=800,
                        justify=tk.LEFT
                    ).pack(padx=10, pady=5, fill=tk.X)
                else:
                    ttk.Label(
                        frame, 
                        text="No objects detected",
                        wraplength=800,
                        justify=tk.LEFT
                    ).pack(padx=10, pady=5, fill=tk.X)
            
            # Region captioning results
            if result.regions and "regions" in self.tasks:
                frame = ttk.LabelFrame(self.scroll_frame, text="Region Captioning")
                frame.pack(fill=tk.X, padx=10, pady=5, expand=True)
                self.task_frames["regions"] = frame
                
                # Display region visualization if available
                if "regions" in result.visualizations:
                    img = result.visualizations["regions"]
                    img.thumbnail((800, 600))
                    photo = ImageTk.PhotoImage(img)
                    img_label = ttk.Label(frame, image=photo)
                    img_label.image = photo
                    img_label.pack(pady=5)
                
                # Display region list
                regions_text = "\n".join([f"• Region {i+1}: {region.get('caption', 'No caption')}" 
                                        for i, region in enumerate(result.regions)])
                if regions_text:
                    ttk.Label(
                        frame, 
                        text=f"Region captions:\n{regions_text}",
                        wraplength=800,
                        justify=tk.LEFT
                    ).pack(padx=10, pady=5, fill=tk.X)
                else:
                    ttk.Label(
                        frame, 
                        text="No region captions available",
                        wraplength=800,
                        justify=tk.LEFT
                    ).pack(padx=10, pady=5, fill=tk.X)
            
            # OCR results
            if result.ocr_text and "ocr" in self.tasks:
                frame = ttk.LabelFrame(self.scroll_frame, text="OCR Results")
                frame.pack(fill=tk.X, padx=10, pady=5, expand=True)
                self.task_frames["ocr"] = frame
                
                ttk.Label(
                    frame, 
                    text=f"Extracted text:\n{result.ocr_text}",
                    wraplength=800,
                    justify=tk.LEFT
                ).pack(padx=10, pady=5, fill=tk.X)
            
            # Phrase grounding results
            if result.grounding_results and "grounding" in self.tasks:
                frame = ttk.LabelFrame(self.scroll_frame, text="Phrase Grounding")
                frame.pack(fill=tk.X, padx=10, pady=5, expand=True)
                self.task_frames["grounding"] = frame
                
                # Display grounding visualization if available
                if "grounding" in result.visualizations:
                    img = result.visualizations["grounding"]
                    img.thumbnail((800, 600))
                    photo = ImageTk.PhotoImage(img)
                    img_label = ttk.Label(frame, image=photo)
                    img_label.image = photo
                    img_label.pack(pady=5)
                
                # Display grounding results
                grounding_text = "\n".join([f"• {phrase}: {details}" 
                                          for phrase, details in result.grounding_results.items()])
                if grounding_text:
                    ttk.Label(
                        frame, 
                        text=f"Grounded phrases:\n{grounding_text}",
                        wraplength=800,
                        justify=tk.LEFT
                    ).pack(padx=10, pady=5, fill=tk.X)
                else:
                    ttk.Label(
                        frame, 
                        text="No phrase grounding results",
                        wraplength=800,
                        justify=tk.LEFT
                    ).pack(padx=10, pady=5, fill=tk.X)
                    
            # Ensure canvas scrollbars are updated
            self.scroll_frame.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            logger.error(f"Error displaying task results: {str(e)}")
            raise

    def next_item(self):
        """Move to next item"""
        if not self.items:
            return
            
        self.current = (self.current + 1) % len(self.items)
        self.show_current()
        
    def prev_item(self):
        """Move to previous item"""
        if not self.items:
            return
            
        self.current = (self.current - 1) % len(self.items)
        self.show_current()

    def move_item(self, dest_dir: Path):
        """Move current item to destination directory with error handling"""
        if not self.items:
            return
            
        try:
            base_name, json_path, img_path = self.items[self.current]
            
            new_img_path = dest_dir / img_path.name
            txt_path = img_path.with_suffix('.txt')
            new_txt_path = dest_dir / txt_path.name
            
            # Create destination directory if needed
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(img_path), str(new_img_path))
            if txt_path.exists():
                shutil.move(str(txt_path), str(new_txt_path))
            
            # Move JSON and other related files
            if json_path.exists():
                data = json.loads(json_path.read_text(encoding='utf-8'))
                data['review_status'] = 'approved' if dest_dir == self.approved_dir else 'rejected'
                data['timestamp'] = datetime.now().isoformat()
                
                new_json_path = dest_dir / f"{base_name}_reviewed.json"
                new_json_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
                json_path.unlink()
            
            # Move visualization files if they exist
            for viz_type in ["objects", "regions", "grounding"]:
                viz_path = img_path.with_suffix(f'.{viz_type}.jpg')
                if viz_path.exists():
                    new_viz_path = dest_dir / viz_path.name
                    shutil.move(str(viz_path), str(new_viz_path))
                
            self.items.pop(self.current)
            if self.items:
                if self.current >= len(self.items):
                    self.current = 0
                self.show_current()
            else:
                self.status_var.set("No more images to review")
                messagebox.showinfo("Info", "All images have been reviewed.")
        except Exception as e:
            logger.error(f"Error moving files: {str(e)}")
            raise

    def approve(self):
        """Approve current item with error handling"""
        try:
            if not self.items:
                return
                
            logger.info(f"Approving item {self.current + 1}/{len(self.items)}")
            self.status_var.set(f"Approving item {self.current + 1}/{len(self.items)}...")
            self.move_item(self.approved_dir)
            self.status_var.set(f"Item approved. {len(self.items)} remaining.")
        except Exception as e:
            logger.error(f"Error approving item: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to approve item: {str(e)}")
        
    def reject(self):
        """Reject current item with error handling"""
        try:
            if not self.items:
                return
                
            logger.info(f"Rejecting item {self.current + 1}/{len(self.items)}")
            self.status_var.set(f"Rejecting item {self.current + 1}/{len(self.items)}...")
            self.move_item(self.rejected_dir)
            self.status_var.set(f"Item rejected. {len(self.items)} remaining.")
        except Exception as e:
            logger.error(f"Error rejecting item: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to reject item: {str(e)}")

    def exit(self):
        """Clean exit of application"""
        self.stop_worker.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        self.root.destroy()


def export_batch_processor(
    input_dir: str, 
    output_dir: str,
    model_name: str = "florence2",
    model_config: Dict = None,
    tasks: List[str] = None,
    recursive: bool = False,
    max_workers: int = 4,
    trigger_word: Optional[str] = None
):
    """
    Export function for batch processing without GUI
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        model_name: Name of model to use
        model_config: Model configuration
        tasks: List of tasks to perform
        recursive: Whether to process subdirectories
        max_workers: Maximum number of parallel workers
        trigger_word: Optional trigger word to add to captions
    """
    try:
        logger.info(f"Starting batch processing with model: {model_name}")
        
        # Initialize model
        model_manager = ModelManager()
        model = model_manager.get_model(model_name, model_config or {})
        
        # Initialize dataset preparator
        dataset_prep = DatasetPreparator()
        
        # Process directory
        if tasks is None:
            tasks = ["caption"]
            
        def callback(path, result, processed, total):
            logger.info(f"Processed {processed}/{total}: {path}")
            return True
            
        # Add trigger word if specified
        if trigger_word:
            def process_callback(path, result, processed, total):
                if result and result.clean_caption:
                    result.clean_caption = f"{trigger_word}, {result.clean_caption}"
                return callback(path, result, processed, total)
        else:
            process_callback = callback
            
        results = dataset_prep.batch_process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            model=model,
            tasks=tasks,
            recursive=recursive,
            max_workers=max_workers,
            callback=process_callback
        )
        
        logger.info(f"Batch processing complete. Processed {len(results)} images.")
        return len(results)
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description='AI Training Dataset Preparation Tool')
    parser.add_argument('--review_dir', default='review', help='Review directory')
    parser.add_argument('--approved_dir', default='approved', help='Approved directory')
    parser.add_argument('--rejected_dir', default='rejected', help='Rejected directory')
    parser.add_argument('--trigger_word', help='Optional trigger word to add to captions')
    parser.add_argument('--model', default='florence2', choices=['florence2', 'janus'],
                      help='Vision model to use (default: florence2)')
    parser.add_argument('--model_size', default='base', choices=['base', 'large', 'instruct'],
                      help='Model size for Florence2 (default: base)')
    parser.add_argument('--batch', action='store_true', help='Run in batch mode without GUI')
    parser.add_argument('--input_dir', help='Input directory for batch mode')
    parser.add_argument('--output_dir', help='Output directory for batch mode')
    parser.add_argument('--tasks', nargs='+', 
                      choices=['caption', 'objects', 'regions', 'ocr', 'grounding'],
                      default=['caption'],
                      help='Tasks to perform in batch mode')
    parser.add_argument('--recursive', action='store_true', 
                      help='Process subdirectories recursively in batch mode')
    parser.add_argument('--max_workers', type=int, default=4,
                      help='Maximum number of parallel workers for batch mode')
    
    try:
        args = parser.parse_args()
        logger.info(f"Starting application with model: {args.model}")
        
        model_config = {"size": args.model_size} if args.model == "florence2" else {}
        
        if args.batch:
            # Batch mode without GUI
            if not args.input_dir or not args.output_dir:
                parser.error("--input_dir and --output_dir are required in batch mode")
                
            export_batch_processor(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                model_name=args.model,
                model_config=model_config,
                tasks=args.tasks,
                recursive=args.recursive,
                max_workers=args.max_workers,
                trigger_word=args.trigger_word
            )
        else:
            # GUI mode
            app = ReviewGUI(
                args.review_dir, 
                args.approved_dir, 
                args.rejected_dir,
                args.trigger_word,
                args.model,
                model_config
            )
            
            app.root.protocol("WM_DELETE_WINDOW", app.exit)
            app.root.mainloop()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()