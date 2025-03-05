# Multi-Vision Toolkit 🖼️ 🤖
 A toolkit for local deployment of state-of-the-art vision models (Florence-2 and BLIP), providing advanced computer vision capabilities including object detection, image captioning, OCR, and visual analysis.

<p align="center">
  <img src="images/Vision-Toolkit_DarkMode_2025-03-05.jpg" width="49%" alt="Vision Toolkit Dark Mode">
  <img src="images/Vision-Toolkit_LightMode_2025-03-05.jpg" width="49%" alt="Vision Toolkit Light Mode">
</p>

## 🚀 Key Features

- **Multiple Vision Models**: Florence-2 (advanced vision tasks) and BLIP (high-quality image captioning)
- **Multi-task Capabilities**: Captioning, object detection, OCR, Visual Question Answering
- **Easy-to-use GUI**: Model switching, image preview, and keyboard shortcuts
- **Dataset Preparation**: Support for AI training dataset creation

## 🛠️ Installation

### Prerequisites
- Python 3.11
- CUDA-capable GPU recommended (8GB VRAM for Florence-2, 4GB for BLIP)

```bash
# Create conda environment
conda create -n vision-env python=3.11
conda activate vision-env

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install core dependencies
pip install -r requirements.txt
```

## 📁 Quick Start

```bash
# Create directories
mkdir -p data/{review,approved,rejected}

# Basic usage
python main.py --review_dir data/review --approved_dir data/approved --rejected_dir data/rejected

# Use specific model
python main.py --review_dir data/review --model florence2  # or --model blip

# Add trigger word to captions
python main.py --review_dir data/review --trigger_word "your_trigger"

# Use Florence-2 model variant
python main.py --review_dir data/review --model florence2 --variant large  # or --variant base
```

## 💻 GUI Features

- **Model Selection**: Switch between Florence-2 and BLIP
- **Image Management**: Preview, approve (A key), or reject (R key)
- **Analysis Display**: View captioning, object detection, and OCR results
- **Metadata Tracking**: Auto-generated JSON and text files

## 📝 Technical Details

### Supported Formats
- Images: `.jpg`, `.jpeg`, `.png`
- Metadata: `.json`, `.txt`

### Model Capabilities

| Model | Capabilities | VRAM Requirements |
|-------|-------------|-------------------|
| Florence-2 (large) | Captioning, object detection, OCR, VQA | 8GB+ |
| Florence-2 (base) | Same as large with lower accuracy | 4-8GB |
| BLIP | High-quality image captioning | 4GB+ |

## 🔧 Troubleshooting

- **Memory Issues**: Use `--variant base` for lower VRAM usage or close other GPU processes
- **Model Loading**: Update transformers with `pip install --upgrade transformers` or clear cache
- **Image Errors**: Verify image format and permissions

## 📄 License

Apache License 2.0
