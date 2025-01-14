# main.py
import argparse
import json
import os
import logging
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import shutil
from datetime import datetime
from typing import Optional, List, Tuple

from models.florence_model import Florence2Model
from models.janus_model import JanusModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetPreparator:
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png'}
        
    def is_supported_image(self, filename: str) -> bool:
        return any(filename.lower().endswith(ext) for ext in self.supported_formats)
        
    def create_caption_file(self, image_path: str, caption: str) -> str:
        try:
            base_path = os.path.splitext(image_path)[0]
            txt_path = f"{base_path}.txt"
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            return txt_path
        except Exception as e:
            logger.error(f"Error creating caption file: {str(e)}")
            raise

class ReviewGUI:
    def __init__(
        self, 
        review_dir: str, 
        approved_dir: str, 
        rejected_dir: str, 
        trigger_word: Optional[str] = None,
        model_name: str = "florence2"
    ):
        logger.info(f"Initializing ReviewGUI with model: {model_name}")
        self.review_dir = review_dir
        self.approved_dir = approved_dir 
        self.rejected_dir = rejected_dir
        self.trigger_word = trigger_word
        
        self.model_name = model_name
        self.model = self._initialize_model(model_name)
        self.dataset_prep = DatasetPreparator()
        
        for dir_path in [review_dir, approved_dir, rejected_dir]:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created/verified directory: {dir_path}")
        
        self.root = tk.Tk()
        self.root.title("AI Training Dataset Preparation Tool")
        self.root.geometry("1200x800")
        
        self.setup_gui()
        self.load_items()

    def _initialize_model(self, model_name: str):
        try:
            if model_name.lower() == "florence2":
                return Florence2Model()
            elif model_name.lower() == "janus":
                return JanusModel()
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {str(e)}")
            raise

    def setup_gui(self):
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
        
        self.img_label = ttk.Label(frame)
        self.img_label.grid(row=1, column=0, pady=10)
        
        self.caption = tk.StringVar()
        caption_label = ttk.Label(
            frame, 
            textvariable=self.caption, 
            wraplength=800,
            justify=tk.LEFT
        )
        caption_label.grid(row=2, column=0, pady=(0, 20))
        
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=3, column=0, pady=10)
        
        approve_btn = ttk.Button(btn_frame, text="Approve (A)", command=self.approve)
        approve_btn.pack(side=tk.LEFT, padx=5)
        
        reject_btn = ttk.Button(btn_frame, text="Reject (R)", command=self.reject)
        reject_btn.pack(side=tk.LEFT, padx=5)
        
        self.root.bind('a', lambda e: self.approve())
        self.root.bind('r', lambda e: self.reject())

    def _on_model_change(self, event):
        try:
            new_model = self.model_var.get()
            if new_model != self.model_name:
                logger.info(f"Switching model from {self.model_name} to {new_model}")
                self.model_name = new_model
                self.model = self._initialize_model(new_model)
                if self.items:
                    self.show_current()
        except Exception as e:
            logger.error(f"Error changing model: {str(e)}")
            raise

    def load_items(self):
        self.items = []
        try:
            if not os.path.exists(self.review_dir):
                logger.warning(f"Review directory not found: {self.review_dir}")
                return
                
            for f in os.listdir(self.review_dir):
                if self.dataset_prep.is_supported_image(f):
                    img_path = os.path.join(self.review_dir, f)
                    base_name = os.path.splitext(f)[0]
                    json_path = os.path.join(self.review_dir, f"{base_name}_for_review.json")
                    
                    if not os.path.exists(json_path):
                        with open(json_path, 'w') as f:
                            json.dump({"results": {"caption": ""}}, f, indent=2)
                    
                    self.items.append((base_name, json_path, img_path))
            
            self.current = 0
            if self.items:
                self.show_current()
            else:
                logger.info("No supported image files found for review.")
        except Exception as e:
            logger.error(f"Error loading items: {str(e)}")
            raise

    def show_current(self):
        if not self.items:
            return
            
        try:
            _, json_path, img_path = self.items[self.current]
            
            description, clean_caption = self.model.analyze_image(img_path)
            
            if self.trigger_word and clean_caption:
                clean_caption = f"{self.trigger_word}, {clean_caption}"
            
            img = Image.open(img_path)
            img.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(img)
            self.img_label.configure(image=photo)
            self.img_label.image = photo
            
            data = {"results": {"caption": description}}
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            if clean_caption:
                self.dataset_prep.create_caption_file(img_path, clean_caption)
            
            self.caption.set(f"Analysis:\n{description}")
            self.root.title(f"Review {self.current + 1}/{len(self.items)}")
        except Exception as e:
            logger.error(f"Error showing current item: {str(e)}")
            raise

    def move_item(self, dest_dir):
        if not self.items:
            return
            
        try:
            base_name, json_path, img_path = self.items[self.current]
            
            img_ext = os.path.splitext(img_path)[1]
            new_img_path = os.path.join(dest_dir, f"{base_name}{img_ext}")
            txt_path = f"{os.path.splitext(img_path)[0]}.txt"
            new_txt_path = os.path.join(dest_dir, f"{base_name}.txt")
            
            shutil.move(img_path, new_img_path)
            if os.path.exists(txt_path):
                shutil.move(txt_path, new_txt_path)
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                data['review_status'] = 'approved' if dest_dir == self.approved_dir else 'rejected'
                data['timestamp'] = datetime.now().isoformat()
                
                new_json_path = os.path.join(dest_dir, f"{base_name}_reviewed.json")
                with open(new_json_path, 'w') as f:
                    json.dump(data, f, indent=2)
                os.remove(json_path)
                
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
        try:
            logger.info(f"Approving item {self.current + 1}/{len(self.items)}")
            self.move_item(self.approved_dir)
        except Exception as e:
            logger.error(f"Error approving item: {str(e)}")
            raise
        
    def reject(self):
        try:
            logger.info(f"Rejecting item {self.current + 1}/{len(self.items)}")
            self.move_item(self.rejected_dir)
        except Exception as e:
            logger.error(f"Error rejecting item: {str(e)}")
            raise

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