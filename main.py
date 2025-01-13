import argparse
import json
import os
from PIL import Image
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk
import shutil
from datetime import datetime

class ReviewGUI:
    def __init__(self, review_dir: str, approved_dir: str, rejected_dir: str):
        # Setup directories
        self.review_dir = review_dir
        self.approved_dir = approved_dir 
        self.rejected_dir = rejected_dir
        for dir_path in [approved_dir, rejected_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize window
        self.root = tk.Tk()
        self.root.title("Image Review")
        self.root.geometry("1200x800")
        
        self.setup_gui()
        self.load_items()
        
    def setup_gui(self):
        # Main layout
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky="nsew")
        
        # Image display
        self.img_label = ttk.Label(frame)
        self.img_label.grid(row=0, column=0, pady=10)
        
        # Caption display
        self.caption = tk.StringVar()
        ttk.Label(frame, textvariable=self.caption, wraplength=800).grid(row=1, column=0)
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, pady=10)
        
        ttk.Button(btn_frame, text="Approve (A)", command=self.approve).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reject (R)", command=self.reject).pack(side=tk.LEFT, padx=5)
        
        # Keyboard shortcuts
        self.root.bind('a', lambda e: self.approve())
        self.root.bind('r', lambda e: self.reject())
        
    def load_items(self):
        self.items = []
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
            
    def show_current(self):
        if self.items:
            _, json_path, img_path = self.items[self.current]
            
            # Show image
            img = Image.open(img_path)
            img.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(img)
            self.img_label.configure(image=photo)
            self.img_label.image = photo
            
            # Show caption
            with open(json_path) as f:
                data = json.load(f)
            caption = data['results'].get('caption', 'No caption')
            self.caption.set(f"Caption: {caption}")
            
            # Update title
            self.root.title(f"Review {self.current + 1}/{len(self.items)}")
            
    def move_item(self, dest_dir):
        if not self.items:
            return
            
        base, json_path, img_path = self.items[self.current]
        
        # Update and save review status
        with open(json_path) as f:
            data = json.load(f)
        data['review_status'] = 'approved' if dest_dir == self.approved_dir else 'rejected'
        data['timestamp'] = datetime.now().isoformat()
        
        # Move files
        shutil.move(img_path, os.path.join(dest_dir, f"{base}_original.png"))
        with open(os.path.join(dest_dir, f"{base}_reviewed.json"), 'w') as f:
            json.dump(data, f, indent=2)
        os.remove(json_path)
        
        # Update display
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
    parser = argparse.ArgumentParser(description='Image Review Tool')
    parser.add_argument('--review_dir', required=True, help='Review directory')
    parser.add_argument('--approved_dir', default='approved', help='Approved directory')
    parser.add_argument('--rejected_dir', default='rejected', help='Rejected directory')
    
    args = parser.parse_args()
    ReviewGUI(args.review_dir, args.approved_dir, args.rejected_dir).root.mainloop()

if __name__ == "__main__":
    main()