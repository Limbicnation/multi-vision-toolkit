# Florence2-vision-toolkit 🖼️ 🤖

A comprehensive Python toolkit for local deployment of Microsoft's Florence-2 vision model. Process images with state-of-the-art computer vision capabilities including object detection, image captioning, and visual analysis.

## 🚀 Key Features
- Local processing of images with Florence-2 model
- Object detection with visualization
- Multi-level image captioning
- Dense region captioning & OCR
- Batch processing support
- Easy-to-use CLI interface

## 🛠️ Quick Start
```python
from florence2_toolkit import Florence2Runner

# Initialize the model
runner = Florence2Runner()

# Process an image
results = runner.predict(image, '<CAPTION>')
