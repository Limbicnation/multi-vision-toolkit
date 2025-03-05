# Multi-Vision Toolkit 🖼️ 🤖

A comprehensive Python toolkit for local deployment of state-of-the-art vision models (Florence-2 and BLIP). Process images with advanced computer vision capabilities including object detection, image captioning, OCR, and visual analysis.

## 🏗️ Model Architectures

### Florence-2 Model Architecture
<img src="images/florence2-toolkit-diagram-florence.jpg" width="1024" alt="Florence-2 Model Architecture">

### BLIP Model Architecture
<img src="images/florence2-toolkit-diagram-janus.jpg" width="1024" alt="BLIP Model Architecture">

## 🚀 Key Features

- Supports multiple vision models:
  - Florence-2: Advanced vision tasks including object detection, OCR, and visual analysis
  - BLIP: High-quality image captioning and visual understanding
- Multi-task capabilities:
  - Image captioning
  - Object detection
  - OCR (Florence-2)
  - Visual Question Answering (Florence-2)
- Batch processing support with cancellation and progress tracking
- Easy-to-use GUI interface with model switching
- **NEW**: Image caching and preloading for faster navigation
- **NEW**: Dark mode UI for extended usage sessions
- **NEW**: Drag-and-drop support for easy image importing
- **NEW**: Export functionality (JSON, CSV, TXT) for analysis results
- **NEW**: 8-bit model quantization for reduced memory usage
- Dataset preparation for AI training
- JSON metadata tracking
- Proper resource management for long sessions

## 🛠️ Installation

### Prerequisites
- Python 3.11
- CUDA-capable GPU (recommended)
- At least 8GB VRAM for Florence-2 (4GB with quantization enabled)
- At least 4GB VRAM for BLIP (2GB with quantization enabled)

### Option 1: Using pip (Recommended)
```bash
# Create conda environment
conda create -n vision-env python=3.11
conda activate vision-env

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install core dependencies
pip install -r requirements.txt
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
pip install timm>=0.9.12 
pip install einops>=0.7.0
pip install python-dotenv>=1.0.0
pip install opencv-python>=4.8.0
pip install matplotlib>=3.5.0
pip install accelerate>=0.25.0
pip install safetensors>=0.4.0

# Install enhanced features dependencies
pip install bitsandbytes>=0.41.0  # For model quantization
pip install tkinterdnd2>=0.3.0    # For drag-and-drop (optional)
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

# Use BLIP
python main.py --review_dir data/review --model blip
```

### With Trigger Word
```bash
python main.py --review_dir data/review --trigger_word "your_trigger" --model florence2
```

### Memory Optimization Options
```bash
# Enable 8-bit quantization for reduced memory usage
python main.py --review_dir data/review --quantize

# Combine with model selection
python main.py --review_dir data/review --model florence2 --variant large --quantize
```

### UI Preferences
```bash
# Start in dark mode
python main.py --review_dir data/review --dark_mode
```

## 💻 GUI Features

- Model switching dropdown
- Image preview with preloading for faster navigation
- Multi-task analysis display
- A/R keys for approve/reject
- Left/Right arrow keys for navigation
- Automatic metadata tracking
- Progress tracking
- Dark mode toggle
- Drag-and-drop support for images
- Export results in multiple formats
- Cancellation support for batch operations

## 📝 Supported Formats

- Images: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`
- Auto-resizing enabled
- Metadata: `.json`, `.txt`
- Export formats: JSON, CSV, TXT

## 🔗 Model Capabilities

### Florence-2
- Advanced image captioning
- Object detection
- OCR
- Visual Question Answering
- Zero-shot learning capabilities
- Supports both base (0.23B) and large (0.77B) variants
- 8-bit quantization support for reduced memory usage

### BLIP
- High-quality image captioning
- Visual understanding
- Robust performance on various image types
- Public model with no authentication required
- 8-bit quantization support for reduced memory usage

## 🔧 Troubleshooting

### Common Issues

1. CUDA/GPU Memory Issues
   - Enable quantization with `--quantize` to reduce VRAM usage by up to 50%
   - For Florence-2 base model: Use `--variant base` for lower VRAM usage
   - Ensure no other GPU processes are running
   - Monitor GPU memory usage with `nvidia-smi`

2. Model Loading Issues
   - Clear transformers cache: `rm -rf ~/.cache/huggingface/`
   - Update transformers: `pip install --upgrade transformers`
   - Check internet connection for model downloads
   - Ensure bitsandbytes is installed if using quantization

3. Image Processing Errors
   - Verify image format support
   - Check image file permissions
   - Ensure sufficient disk space

## 📈 Performance Notes

- Florence-2 large (0.77B): Best for complex visual tasks
- Florence-2 base (0.23B): Good balance of performance and resource usage
- BLIP: Excellent for image captioning, lower resource requirements
- With quantization: ~50% reduction in memory usage with minimal quality loss
- Image caching dramatically improves navigation speed for large datasets

## 📄 License

Apache License 2.0

## 🤝 Contributing

Pull requests welcome! See CONTRIBUTING.md for guidelines.
