# Florence2-vision-toolkit 🖼️ 🤖

A comprehensive Python toolkit for local deployment of Microsoft's Florence-2 vision model. Process images with state-of-the-art computer vision capabilities including object detection, image captioning, and visual analysis.

## 🚀 Key Features
- Local processing of images with Florence-2 model
- Object detection with visualization
- Multi-level image captioning
- Dense region captioning & OCR
- Batch processing support
- Easy-to-use GUI interface for review
- Dataset preparation for AI training

## 🛠️ Quick Start

### 1. Create Conda Environment
```bash
# Create new conda environment with Python 3.11
conda create -n florence2-env python=3.11
conda activate florence2-env

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install torch transformers pillow einops timm
```

### 2. Setup Project Structure
```bash
# Create necessary directories
mkdir -p data/{review,approved,rejected}
```

### 3. Prepare Dataset
Place your images in the review directory:
- Supported formats: `.jpg`, `.jpeg`, `.png`
- Images will be automatically resized during processing
- No manual cropping required
- Webp format is not supported

#### Option A: Automatic Caption Generation
Simply place images in the review directory:
```bash
cp your_images/*.jpg data/review/
```
The tool will automatically:
- Generate captions using Florence-2
- Create matching .txt files
- Move processed files to approved/rejected folders

#### Option B: Manual Caption Setup
For each image, you can optionally create:
1. A matching .txt file with the same name
2. A JSON metadata file (automatically created if missing)

Example structure:
```
data/review/
├── image1.jpg
├── image1.txt          # Optional: Custom caption
└── image1_for_review.json  # Created automatically
```

### 4. Run the Review GUI
```bash
# Basic usage
python main.py --review_dir data/review --approved_dir data/approved --rejected_dir data/rejected

# With trigger word
python main.py --review_dir data/review --approved_dir data/approved --rejected_dir data/rejected --trigger_word "your_trigger"
```

## 💻 GUI Usage
- View images and generated captions
- `A` key: Approve and move to approved directory
- `R` key: Reject and move to rejected directory
- Automatic caption file generation
- JSON metadata tracking

## 📁 Directory Structure
```
project_root/
├── data/
│   ├── review/      # Place images here
│   ├── approved/    # Approved items
│   └── rejected/    # Rejected items
└── main.py         # GUI implementation
```

## 🔧 Requirements
- Python 3.11
- PyTorch 2.0.0+
- transformers 4.36.0+
- Pillow, einops, timm

## 📄 License
Apache License 2.0 - see LICENSE file for details.