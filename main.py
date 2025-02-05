import argparse
import json
import os
import logging
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox
import shutil
from datetime import datetime
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import torch

from models.florence_model import Florence2Model
from models.janus_model import JanusModel

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

class ModelManager:
    """Manages model initialization and switching"""
    def __init__(self):
        self.models: Dict[str, object] = {}
        self._current_model = None
        self._current_model_name = None
        
    def initialize_model(self, model_name: str) -> object:
        """Initialize a model with error handling and caching"""
        try:
            if model_name in self.models:
                logger.info(f"Using cached model: {model_name}")
                return self.models[model_name]
            
            logger.info(f"Initializing new model: {model_name}")
            if model_name.lower() == "florence2":
                model = Florence2Model()
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
            
            self.models[model_name] = model
            return model
            
        except Exception as e:
            logger.error(f"Error initializing model {model_name}: {str(e)}")
            raise

    def get_model(self, model_name: str) -> object:
        """Get a model, initializing if necessary"""
        if self._current_model_name != model_name:
            self._current_model = self.initialize_model(model_name)
            self._current_model_name = model_name
        return self._current_model

class DatasetPreparator:
    """Handles dataset preparation and file operations"""
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png'}
        
    def is_supported_image(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self.supported_formats
        
    def create_caption_file(self, image_path: str, caption: str) -> str:
        try:
            txt_path = Path(image_path).with_suffix('.txt')
            txt_path.write_text(caption, encoding='utf-8')
            return str(txt_path)
        except Exception as e:
            logger.error(f"Error creating caption file: {str(e)}")
            raise

class ReviewGUI:
    """Main GUI application for reviewing images"""
    def __init__(
        self, 
        review_dir: str, 
        approved_dir: str, 
        rejected_dir: str, 
        trigger_word: Optional[str] = None,
        model_name: str = "florence2"
    ):
        logger.info(f"Initializing ReviewGUI with model: {model_name}")
        self.review_dir = Path(review_dir)
        self.approved_dir = Path(approved_dir)
        self.rejected_dir = Path(rejected_dir)
        self.trigger_word = trigger_word
        
        self.model_name = model_name
        self.model_manager = ModelManager()
        try:
            self.model = self.model_manager.get_model(model_name)
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize model {model_name}. Falling back to Florence2.")
            self.model_name = "florence2"
            self.model = self.model_manager.get_model("florence2")
        
        self.dataset_prep = DatasetPreparator()
        
        # Create directories
        for dir_path in [self.review_dir, self.approved_dir, self.rejected_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created/verified directory: {dir_path}")
        
        self.root = tk.Tk()
        self.root.title("AI Training Dataset Preparation Tool")
        self.root.geometry("1200x800")
        
        self.setup_gui()
        self.load_items()

    def setup_gui(self):
        """Setup GUI components with error handling"""
        try:
            frame = ttk.Frame(self.root, padding="10")
            frame.grid(row=0, column=0, sticky="nsew")
            
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            
            # Model selection
            model_frame = ttk.Frame(frame)
            model_frame.grid(row=0, column=0, pady=5)
            
            ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT, padx=5)
            self.model_var = tk.StringVar(value=self.model_name)
            model_combo = ttk.Combobox(
                model_frame, 
                textvariable=self.model_var,
                values=["florence2", "janus"],
                state="readonly"
            )
            model_combo.pack(side=tk.LEFT, padx=5)
            model_combo.bind('<<ComboboxSelected>>', self._on_model_change)
            
            # Image display
            self.img_label = ttk.Label(frame)
            self.img_label.grid(row=1, column=0, pady=10)
            
            # Caption display
            self.caption = tk.StringVar()
            caption_label = ttk.Label(
                frame, 
                textvariable=self.caption, 
                wraplength=800,
                justify=tk.LEFT
            )
            caption_label.grid(row=2, column=0, pady=(0, 20))
            
            # Control buttons
            btn_frame = ttk.Frame(frame)
            btn_frame.grid(row=3, column=0, pady=10)
            
            approve_btn = ttk.Button(btn_frame, text="Approve (A)", command=self.approve)
            approve_btn.pack(side=tk.LEFT, padx=5)
            
            reject_btn = ttk.Button(btn_frame, text="Reject (R)", command=self.reject)
            reject_btn.pack(side=tk.LEFT, padx=5)
            
            # Keyboard shortcuts
            self.root.bind('a', lambda e: self.approve())
            self.root.bind('r', lambda e: self.reject())
            
        except Exception as e:
            logger.error(f"Error setting up GUI: {str(e)}")
            raise

    def _on_model_change(self, event):
        """Handle model switching with error handling"""
        try:
            new_model = self.model_var.get()
            if new_model != self.model_name:
                logger.info(f"Switching model from {self.model_name} to {new_model}")
                try:
                    self.model = self.model_manager.get_model(new_model)
                    self.model_name = new_model
                    if self.items:
                        self.show_current()
                except Exception as e:
                    logger.error(f"Failed to switch to model {new_model}: {str(e)}")
                    messagebox.showerror("Error", f"Failed to switch to {new_model}. Reverting to previous model.")
                    self.model_var.set(self.model_name)
        except Exception as e:
            logger.error(f"Error in model change handler: {str(e)}")
            raise

    def load_items(self):
        """Load image items with error handling"""
        self.items = []
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
            
            self.current = 0
            if self.items:
                self.show_current()
            else:
                logger.info("No supported image files found for review.")
                messagebox.showinfo("Info", "No images found for review.")
        except Exception as e:
            logger.error(f"Error loading items: {str(e)}")
            raise

    def show_current(self):
        """Display current image with error handling"""
        if not self.items:
            return
            
        try:
            _, json_path, img_path = self.items[self.current]
            
            try:
                description, clean_caption = self.model.analyze_image(str(img_path))
            except Exception as e:
                logger.error(f"Error analyzing image: {str(e)}")
                description = "Error analyzing image"
                clean_caption = None
                messagebox.showwarning("Warning", f"Error analyzing image: {str(e)}")
            
            if self.trigger_word and clean_caption:
                clean_caption = f"{self.trigger_word}, {clean_caption}"
            
            # Load and display image
            img = Image.open(img_path)
            img.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(img)
            self.img_label.configure(image=photo)
            self.img_label.image = photo
            
            # Save analysis results
            data = {"results": {"caption": description}}
            json_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
            
            if clean_caption:
                self.dataset_prep.create_caption_file(str(img_path), clean_caption)
            
            self.caption.set(f"Analysis:\n{description}")
            self.root.title(f"Review {self.current + 1}/{len(self.items)}")
        except Exception as e:
            logger.error(f"Error showing current item: {str(e)}")
            raise

    def move_item(self, dest_dir: Path):
        """Move current item to destination directory with error handling"""
        if not self.items:
            return
            
        try:
            base_name, json_path, img_path = self.items[self.current]
            
            new_img_path = dest_dir / img_path.name
            txt_path = img_path.with_suffix('.txt')
            new_txt_path = dest_dir / txt_path.name
            
            shutil.move(str(img_path), str(new_img_path))
            if txt_path.exists():
                shutil.move(str(txt_path), str(new_txt_path))
            
            if json_path.exists():
                data = json.loads(json_path.read_text(encoding='utf-8'))
                data['review_status'] = 'approved' if dest_dir == self.approved_dir else 'rejected'
                data['timestamp'] = datetime.now().isoformat()
                
                new_json_path = dest_dir / f"{base_name}_reviewed.json"
                new_json_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
                json_path.unlink()
                
            self.items.pop(self.current)
            if self.items:
                if self.current >= len(self.items):
                    self.current = 0
                self.show_current()
            else:
                self.root.quit()
        except Exception as e:
            logger.error(f"Error moving files: {str(e)}")
            raise

    def approve(self):
        """Approve current item with error handling"""
        try:
            logger.info(f"Approving item {self.current + 1}/{len(self.items)}")
            self.move_item(self.approved_dir)
        except Exception as e:
            logger.error(f"Error approving item: {str(e)}")
            messagebox.showerror("Error", f"Failed to approve item: {str(e)}")
        
    def reject(self):
        """Reject current item with error handling"""
        try:
            logger.info(f"Rejecting item {self.current + 1}/{len(self.items)}")
            self.move_item(self.rejected_dir)
        except Exception as e:
            logger.error(f"Error rejecting item: {str(e)}")
            messagebox.showerror("Error", f"Failed to reject item: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='AI Training Dataset Preparation Tool')
    parser.add_argument('--review_dir', required=True, help='Review directory')
    parser.add_argument('--approved_dir', default='approved', help='Approved directory')
    parser.add_argument('--rejected_dir', default='rejected', help='Rejected directory')
    parser.add_argument('--trigger_word', help='Optional trigger word to add to captions')
    parser.add_argument('--model', default='florence2', choices=['florence2', 'janus'],
                      help='Vision model to use (default: florence2)')
    
    try:
        args = parser.parse_args()
        logger.info(f"Starting application with model: {args.model}")
        
        app = ReviewGUI(
            args.review_dir, 
            args.approved_dir, 
            args.rejected_dir,
            args.trigger_word,
            args.model
        )
        app.root.mainloop()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()