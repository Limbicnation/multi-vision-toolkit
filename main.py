import argparse
import json
import os
from PIL import Image, ImageStat
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk
import shutil
from datetime import datetime
import colorsys
import numpy as np
from collections import Counter

class ImageAnalyzer:
    def __init__(self):
        self.color_names = {
            'red': ((340, 360), (0, 20)),
            'green': ((90, 150),),
            'blue': ((210, 270),),
            'yellow': ((50, 70),),
            'purple': ((270, 340),),
            'orange': ((20, 50),),
            'brown': ((0, 50),),  # With low saturation/value
            'gold': ((40, 50),)   # With high saturation/value
        }

    def get_dominant_colors(self, image, num_colors=3):
        # Convert image to RGB if it isn't
        img = image.convert('RGB')
        # Resize image to speed up processing
        img = img.resize((150, 150))
        
        # Get colors from image
        pixels = np.float32(img).reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, num_colors, None, criteria, 10, flags)
        
        # Sort by frequency
        _, counts = np.unique(labels, return_counts=True)
        colors = palette[np.argsort(-counts)]
        
        # Convert to HSV for better color naming
        colors_hsv = []
        for color in colors:
            rgb_normalized = tuple(c/255 for c in color)
            hsv = colorsys.rgb_to_hsv(*rgb_normalized)
            colors_hsv.append(hsv)
            
        return colors_hsv

    def analyze_brightness(self, image):
        stat = ImageStat.Stat(image)
        brightness = sum(stat.mean) / len(stat.mean)  # Average brightness
        if brightness < 85:
            return "dark"
        elif brightness > 170:
            return "bright"
        return "moderate"

    def get_color_name(self, hsv):
        h, s, v = hsv
        h = h * 360  # Convert to 360 scale
        
        if s < 0.15:
            if v < 0.3:
                return "black"
            elif v > 0.8:
                return "white"
            return "gray"
            
        if s < 0.35 and v < 0.6:
            return "brown"
            
        for name, ranges in self.color_names.items():
            for range_tuple in ranges:
                if range_tuple[0] <= h <= range_tuple[1]:
                    if name == "gold" and s > 0.5 and v > 0.6:
                        return "gold"
                    elif name != "gold":
                        return name
        
        return "unknown"

    def analyze_image(self, image_path):
        img = Image.open(image_path)
        
        # Get image characteristics
        brightness = self.analyze_brightness(img)
        dominant_colors = self.get_dominant_colors(img)
        color_names = [self.get_color_name(color) for color in dominant_colors]
        
        # Filter out duplicates and unknowns while preserving order
        color_names = list(dict.fromkeys([c for c in color_names if c != "unknown"]))
        
        # Build description
        desc_parts = []
        
        # Describe the overall tone
        desc_parts.append(f"A {brightness} toned artistic portrait")
        
        # Describe dominant colors
        if color_names:
            if "gold" in color_names:
                desc_parts.append("with striking gold accents")
            color_desc = " and ".join(color_names[:2])
            desc_parts.append(f"featuring {color_desc} elements")
            
        if any(c in color_names for c in ["green"]):
            desc_parts.append("with distinctive green highlights")
            
        desc_parts.append("showcasing surreal and ethereal qualities")
        
        return " ".join(desc_parts)

class ReviewGUI:
    def __init__(self, review_dir: str, approved_dir: str, rejected_dir: str):
        # Setup directories and analyzer
        self.review_dir = review_dir
        self.approved_dir = approved_dir 
        self.rejected_dir = rejected_dir
        self.image_analyzer = ImageAnalyzer()
        
        # Create all required directories
        for dir_path in [review_dir, approved_dir, rejected_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize window
        self.root = tk.Tk()
        self.root.title("Image Review")
        self.root.geometry("1200x800")
        
        self.setup_gui()
        self.load_items()
        
    def setup_gui(self):
        # [Previous GUI setup code remains the same]
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky="nsew")
        
        self.img_label = ttk.Label(frame)
        self.img_label.grid(row=0, column=0, pady=10)
        
        self.caption = tk.StringVar()
        ttk.Label(frame, textvariable=self.caption, wraplength=800).grid(row=1, column=0)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, pady=10)
        
        ttk.Button(btn_frame, text="Approve (A)", command=self.approve).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reject (R)", command=self.reject).pack(side=tk.LEFT, padx=5)
        
        self.root.bind('a', lambda e: self.approve())
        self.root.bind('r', lambda e: self.reject())

    def load_items(self):
        self.items = []
        if not os.path.exists(self.review_dir):
            print(f"Review directory not found: {self.review_dir}")
            return
            
        for f in os.listdir(self.review_dir):
            if f.endswith('_for_review.json'):
                base = f.replace('_for_review.json', '')
                json_path = os.path.join(self.review_dir, f)
                img_path = os.path.join(self.review_dir, f"{base}_original.png")
                
                if os.path.exists(img_path):
                    self.items.append((base, json_path, img_path))
        
        self.current = 0
        if self.items:
            self.show_current()
        else:
            print("No items found for review. Please add files with '_for_review.json' and '_original.png' suffixes.")
            
    def show_current(self):
        if self.items:
            _, json_path, img_path = self.items[self.current]
            
            # Generate caption using image analyzer
            caption = self.image_analyzer.analyze_image(img_path)
            
            # Show image
            img = Image.open(img_path)
            img.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(img)
            self.img_label.configure(image=photo)
            self.img_label.image = photo
            
            # Update JSON with new caption
            with open(json_path) as f:
                data = json.load(f)
            data['results']['caption'] = caption
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.caption.set(f"Caption: {caption}")
            self.root.title(f"Review {self.current + 1}/{len(self.items)}")
            
    def move_item(self, dest_dir):
        # [Previous move_item code remains the same]
        if not self.items:
            return
            
        base, json_path, img_path = self.items[self.current]
        
        with open(json_path) as f:
            data = json.load(f)
        data['review_status'] = 'approved' if dest_dir == self.approved_dir else 'rejected'
        data['timestamp'] = datetime.now().isoformat()
        
        shutil.move(img_path, os.path.join(dest_dir, f"{base}_original.png"))
        with open(os.path.join(dest_dir, f"{base}_reviewed.json"), 'w') as f:
            json.dump(data, f, indent=2)
        os.remove(json_path)
        
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
    parser = argparse.ArgumentParser(description='Enhanced Image Review Tool')
    parser.add_argument('--review_dir', required=True, help='Review directory')
    parser.add_argument('--approved_dir', default='approved', help='Approved directory')
    parser.add_argument('--rejected_dir', default='rejected', help='Rejected directory')
    
    args = parser.parse_args()
    ReviewGUI(args.review_dir, args.approved_dir, args.rejected_dir).root.mainloop()

if __name__ == "__main__":
    main()