# main.py
import argparse
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
import os
import numpy as np
from typing import Dict, List, Union
import random
import requests
from pathlib import Path

class Florence2Runner:
    def __init__(self, model_path: str = 'microsoft/Florence-2-large'):
        """Initialize Florence-2 model and processor"""
        logging.info(f"Initializing Florence-2 with model: {model_path}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
    def predict(self, image: Image.Image, task_prompt: str, text_input: str = None) -> Dict:
        """Run prediction with Florence-2"""
        prompt = task_prompt if text_input is None else task_prompt + text_input
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device, torch.float16 if self.device == 'cuda' else torch.float32)
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )[0]
        
        return self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )

    def visualize_bboxes(self, image: Image.Image, data: Dict, 
                        save_path: str = None) -> None:
        """Visualize bounding boxes on image"""
        fig, ax = plt.subplots()
        ax.imshow(image)
        
        for bbox, label in zip(data['bboxes'], data['labels']):
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)
            plt.text(
                x1, y1,
                label,
                color='white',
                fontsize=8,
                bbox=dict(facecolor='red', alpha=0.5)
            )
        
        ax.axis('off')
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """Process all images in a directory"""
        os.makedirs(output_dir, exist_ok=True)
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        for img_file in image_files:
            image_path = os.path.join(input_dir, img_file)
            image = Image.open(image_path)
            
            # Run different tasks
            results = {}
            
            # Basic caption
            results['caption'] = self.predict(image, '<CAPTION>')
            
            # Object detection
            od_results = self.predict(image, '<OD>')
            results['object_detection'] = od_results
            
            # Save visualizations
            base_name = os.path.splitext(img_file)[0]
            bbox_save_path = os.path.join(output_dir, f"{base_name}_boxes.png")
            self.visualize_bboxes(
                image,
                od_results['<OD>'],
                save_path=bbox_save_path
            )
            
            # Save results
            import json
            results_path = os.path.join(output_dir, f"{base_name}_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Florence-2 Local Runner')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Directory to save outputs')
    parser.add_argument('--model_path', type=str,
                      default='microsoft/Florence-2-large',
                      help='Florence-2 model path or identifier')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run
    runner = Florence2Runner(args.model_path)
    runner.process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()