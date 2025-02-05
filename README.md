# Multi-Vision Toolkit 🖼️ 🤖

A comprehensive Python toolkit for local deployment of state-of-the-art vision models (Florence-2 and Janus-Pro). Process images with advanced computer vision capabilities including object detection, image captioning, and visual analysis.

## 🚀 Key Features
- Supports multiple vision models:
  - Florence-2: Object detection & visual analysis
  - Janus-Pro-1B: Enhanced image captioning
- Batch processing support
- Easy-to-use GUI interface with model switching
- Dataset preparation for AI training
- JSON metadata tracking

## 🛠️ Installation

### Prerequisites
- Python 3.11
- CUDA-capable GPU (recommended)
- At least 8GB VRAM for Florence-2
- At least 4GB VRAM for Janus-Pro-1B

### Option 1: Using pip (Recommended)
```bash
# Create conda environment
conda create -n vision-env python=3.11
conda activate vision-env

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install vision dependencies
pip install timm einops

# Install remaining dependencies
pip install -r requirements.txt

# Update transformers (if needed for Janus-Pro)
pip install --upgrade transformers
# or
pip install git+https://github.com/huggingface/transformers.git
```

### Option 2: Manual Installation
```bash
# Create conda environment
conda create -n vision-env python=3.11
conda activate vision-env

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install core dependencies
pip install transformers>=4.36.0 
pip install Pillow>=9.0.0 
pip install matplotlib>=3.5.0 
pip install opencv-python 
pip install accelerate 
pip install safetensors
pip install timm einops
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

## 🔧 Troubleshooting

### Common Issues
1. `ModuleNotFoundError: No module named 'timm'` or `'einops'`
   ```bash
   pip install timm einops
   ```

2. Transformers version issues with Janus-Pro
   ```bash
   pip install --upgrade transformers
   # or
   pip install git+https://github.com/huggingface/transformers.git
   ```

3. CUDA/GPU issues
   - Ensure NVIDIA drivers are up to date
   - Check CUDA compatibility with PyTorch version

## 📄 License
Apache License 2.0

## 🤝 Contributing
Pull requests welcome! See CONTRIBUTING.md for guidelines.