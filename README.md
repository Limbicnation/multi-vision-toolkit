# Multi-Vision Toolkit 🖼️ 🤖

A comprehensive Python toolkit for local deployment of state-of-the-art vision models (Florence-2 and Janus-Pro). Process images with advanced computer vision capabilities including object detection, image captioning, and visual analysis.

## 🚀 Key Features
- Supports multiple vision models:
  - Florence-2: Object detection & visual analysis
  - Janus-Pro-7B: Enhanced image captioning
- Batch processing support
- Easy-to-use GUI interface with model switching
- Dataset preparation for AI training
- JSON metadata tracking

## 🛠️ Installation

### Option 1: Using pip (Recommended)
```bash
# Create conda environment
conda create -n vision-env python=3.11
conda activate vision-env

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Manual Installation
```bash
# Create conda environment
conda create -n vision-env python=3.11
conda activate vision-env

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install transformers>=4.36.0 Pillow>=9.0.0 matplotlib>=3.5.0 opencv-python accelerate safetensors
```

## 📁 Project Setup
```bash
# Create directories
mkdir -p data/{review,approved,rejected}
```

## 🎯 Usage

### Basic Usage
```bash
python main.py --review_dir data/review --approved_dir data/approved --rejected_dir data/rejected
```

### With Model Selection
```bash
# Use Florence-2 (default)
python main.py --review_dir data/review --model florence2

# Use Janus-Pro
python main.py --review_dir data/review --model janus
```

### With Trigger Word
```bash
python main.py --review_dir data/review --trigger_word "your_trigger" --model janus
```

## 💻 GUI Features
- Model switching dropdown
- Image preview
- Caption generation
- A/R keys for approve/reject
- Automatic metadata tracking

## 📝 Supported Formats
- Images: `.jpg`, `.jpeg`, `.png`
- Auto-resizing enabled
- Metadata: `.json`, `.txt`

## 📄 License
Apache License 2.0

## 🤝 Contributing
Pull requests welcome! See CONTRIBUTING.md for guidelines.