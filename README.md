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
```

## 🔧 Implementation Steps

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Create Test Directories
```bash
mkdir -p test_images/input test_images/output
```

### 3. Prepare Test Data
```bash
# Copy some test images to input folder
cp path/to/your/images/*.jpg test_images/input/
```

### 4. Run the Script
```bash
python main.py \
    --model_path "florence-2" \
    --input_dir "test_images/input" \
    --output_dir "test_images/output"
```

### 5. Check Results
```bash
ls -l test_images/output/
```

The script will:
- Load all images from `test_images/input`
- Process each image through Florence-2 model
- Save results to `test_images/output`
- Log progress in console

### Output Structure
```
test_images/output/
├── image1_boxes.png        # Visualization of detected objects
├── image1_results.json     # Detailed analysis results
├── image2_boxes.png
└── image2_results.json
```

## 💡 Use Cases
- Automated image analysis
- Content tagging and organization
- Visual search applications
- Research and development
- Dataset creation for ML/AI

## 📦 Installation
```bash
pip install florence2-vision-toolkit
```

## 📄 License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
