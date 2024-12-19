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

# Implementation Steps

1. Create Test Directories

```mkdir -p test_images/input test_images/output```

2. Copy Some Test Images

```# Copy some test images to input folder
cp path/to/your/images/*.jpg test_images/input/```

3. Run the Script

```python main.py \
    --model_path "florence-2" \
    --input_dir "test_images/input" \
    --output_dir "test_images/output"```


The script will:

Load all images from test_images/input
Process each image through Florence-2 model
Save results to test_images/output
Log progress in console

To check results:

```ls -l test_images/output/```
