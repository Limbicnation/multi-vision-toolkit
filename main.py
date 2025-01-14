import argparse
import json
import os
from PIL import Image, ImageStat
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk
import shutil
from datetime import datetime
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

class DatasetPreparator:
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png'}
        
    def is_supported_image(self, filename):
        return any(filename.lower().endswith(ext) for ext in self.supported_formats)
        
    def create_caption_file(self, image_path, caption):
        """Create a .txt file with the same name as the image containing the caption"""
        base_path = os.path.splitext(image_path)[0]
        txt_path = f"{base_path}.txt"
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(caption)
        return txt_path

class FlorenceAnalyzer:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print("Loading Florence-2 model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", 
            torch_dtype=self.torch_dtype, 
            trust_remote_code=True
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", 
            trust_remote_code=True
        )
        print("Model loaded successfully!")

    def clean_output(self, text):
        """Clean model output by removing special tokens and HTML tags"""
        import re
        
        # Remove special tokens
        text = text.replace('</s>', '').replace('<s>', '')
        
        # Remove HTML-like tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove locations tags
        text = re.sub(r'<loc_\d+>', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        return text.strip()

    def analyze_image(self, image_path):
        try:
            # Load image
            image = Image.open(image_path)
            
            # Process with Florence-2 for caption
            inputs = self.processor(
                text="<image>Describe this image in detail:",
                images=image,
                return_tensors="pt"
            ).to(self.device, self.torch_dtype)

            # Generate caption
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
                do_sample=False
            )
            
            caption = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Process for object detection
            inputs_od = self.processor(
                text="<OD>",
                images=image,
                return_tensors="pt"
            ).to(self.device, self.torch_dtype)

            generated_ids_od = self.model.generate(
                input_ids=inputs_od["input_ids"],
                pixel_values=inputs_od["pixel_values"],
                max_new_tokens=128,
                num_beams=3,
                do_sample=False
            )
            
            objects_text = self.processor.post_process_generation(
                self.processor.batch_decode(generated_ids_od, skip_special_tokens=False)[0],
                task="<OD>",
                image_size=(image.width, image.height)
            )
            
            # Clean outputs
            caption = self.clean_output(caption)
            objects = [self.clean_output(obj) for obj in objects_text if obj.strip()]
            
            # Format final description
            description = (
                f"Description: {caption}\n\n"
                f"Detected objects: {', '.join(objects)}"
            )
            
            return description, caption  # Return both full description and clean caption
                
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return "Error analyzing image. Please check the console for details.", ""

class ReviewGUI:
    def __init__(self, review_dir: str, approved_dir: str, rejected_dir: str, trigger_word: str = None):
        # Setup directories and analyzers
        self.review_dir = review_dir
        self.approved_dir = approved_dir 
        self.rejected_dir = rejected_dir
        self.trigger_word = trigger_word
        self.analyzer = FlorenceAnalyzer()
        self.dataset_prep = DatasetPreparator()
        
        # Create required directories
        for dir_path in [review_dir, approved_dir, rejected_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize window
        self.root = tk.Tk()
        self.root.title("AI Training Dataset Preparation Tool")
        self.root.geometry("1200x800")
        
        self.setup_gui()
        self.load_items()

    def setup_gui(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        
        # Image display
        self.img_label = ttk.Label(frame)
        self.img_label.grid(row=0, column=0, pady=10)
        
        # Caption display and edit
        self.caption = tk.StringVar()
        caption_label = ttk.Label(
            frame, 
            textvariable=self.caption, 
            wraplength=800,
            justify=tk.LEFT
        )
        caption_label.grid(row=1, column=0, pady=(0, 20))
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, pady=10)
        
        approve_btn = ttk.Button(btn_frame, text="Approve (A)", command=self.approve)
        approve_btn.pack(side=tk.LEFT, padx=5)
        
        reject_btn = ttk.Button(btn_frame, text="Reject (R)", command=self.reject)
        reject_btn.pack(side=tk.LEFT, padx=5)
        
        self.root.bind('a', lambda e: self.approve())
        self.root.bind('r', lambda e: self.reject())

    def load_items(self):
        self.items = []
        if not os.path.exists(self.review_dir):
            print(f"Review directory not found: {self.review_dir}")
            return
            
        for f in os.listdir(self.review_dir):
            if self.dataset_prep.is_supported_image(f):
                img_path = os.path.join(self.review_dir, f)
                base_name = os.path.splitext(f)[0]
                json_path = os.path.join(self.review_dir, f"{base_name}_for_review.json")
                
                # Create JSON if it doesn't exist
                if not os.path.exists(json_path):
                    with open(json_path, 'w') as f:
                        json.dump({"results": {"caption": ""}}, f, indent=2)
                
                self.items.append((base_name, json_path, img_path))
        
        self.current = 0
        if self.items:
            self.show_current()
        else:
            print("No supported image files found for review.")

    def show_current(self):
        if self.items:
            _, json_path, img_path = self.items[self.current]
            
            # Generate caption using Florence-2
            description, clean_caption = self.analyzer.analyze_image(img_path)
            
            # Add trigger word if specified
            if self.trigger_word and clean_caption:
                clean_caption = f"[trigger] {clean_caption}"
                clean_caption = clean_caption.replace("[trigger]", self.trigger_word)
            
            # Show image
            img = Image.open(img_path)
            img.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(img)
            self.img_label.configure(image=photo)
            self.img_label.image = photo
            
            # Update JSON and create caption file
            data = {"results": {"caption": description}}
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Create .txt caption file
            if clean_caption:
                self.dataset_prep.create_caption_file(img_path, clean_caption)
            
            self.caption.set(f"Analysis:\n{description}")
            self.root.title(f"Review {self.current + 1}/{len(self.items)}")

    def move_item(self, dest_dir):
        if not self.items:
            return
            
        base_name, json_path, img_path = self.items[self.current]
        
        try:
            # Move image and its caption file
            img_ext = os.path.splitext(img_path)[1]
            new_img_path = os.path.join(dest_dir, f"{base_name}{img_ext}")
            txt_path = f"{os.path.splitext(img_path)[0]}.txt"
            new_txt_path = os.path.join(dest_dir, f"{base_name}.txt")
            
            shutil.move(img_path, new_img_path)
            if os.path.exists(txt_path):
                shutil.move(txt_path, new_txt_path)
            
            # Handle JSON
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                data['review_status'] = 'approved' if dest_dir == self.approved_dir else 'rejected'
                data['timestamp'] = datetime.now().isoformat()
                
                new_json_path = os.path.join(dest_dir, f"{base_name}_reviewed.json")
                with open(new_json_path, 'w') as f:
                    json.dump(data, f, indent=2)
                os.remove(json_path)
                
        except Exception as e:
            print(f"Error moving files: {str(e)}")
            return
            
        self.items.pop(self.current)
        if self.items:
            if self.current >= len(self.items):
                self.current = 0
            self.show_current()
        else:
            self.root.quit()

    def approve(self):
        self.move_item(self.approved_dir)
        
    def reject(self):
        self.move_item(self.rejected_dir)

def main():
    parser = argparse.ArgumentParser(description='AI Training Dataset Preparation Tool')
    parser.add_argument('--review_dir', required=True, help='Review directory')
    parser.add_argument('--approved_dir', default='approved', help='Approved directory')
    parser.add_argument('--rejected_dir', default='rejected', help='Rejected directory')
    parser.add_argument('--trigger_word', help='Optional trigger word to add to captions')
    
    args = parser.parse_args()
    ReviewGUI(
        args.review_dir, 
        args.approved_dir, 
        args.rejected_dir,
        args.trigger_word
    ).root.mainloop()

if __name__ == "__main__":
    main()