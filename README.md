# Multi-Vision Toolkit 🖼️ 🤖
 A toolkit for local deployment of state-of-the-art vision models (Florence-2, BLIP, and Qwen2.5-VL), providing advanced computer vision capabilities including object detection, image captioning, OCR, and visual analysis.

<p align="center">
  <img src="images/Vision-Toolkit_DarkMode_2025-03-05.jpg" width="49%" alt="Vision Toolkit Dark Mode">
  <img src="images/Vision-Toolkit_LightMode_2025-03-05.jpg" width="49%" alt="Vision Toolkit Light Mode">
</p>

## 🚀 Key Features

- **Multiple Vision Models**: Florence-2 (advanced vision tasks), BLIP (high-quality image captioning), and Qwen2.5-VL (high-quality multimodal captioning)
- **Multi-task Capabilities**: Captioning, object detection, OCR, Visual Question Answering
- **Easy-to-use GUI**: Model switching, image preview, and keyboard shortcuts
- **Dataset Preparation**: Support for AI training dataset creation
- **Quality Controls**: Generate captions in standard, detailed, or creative modes
- **Drag and Drop**: Easily process images or entire folders by dragging them directly into the application
- **Batch Processing**: Process multiple images at once with progress tracking
- **Export Functionality**: Export analysis results to CSV or JSON formats
- **Image Caching**: Faster navigation with preloading and caching of image analyses
- **Auto-download Models**: Models are automatically downloaded when needed and cached for future use

## 🛠️ Installation

### Prerequisites
- Python 3.11
- CUDA-capable GPU recommended (Check VRAM requirements below)

```bash
# Create conda environment
conda create -n vision-env python=3.11
conda activate vision-env

# Install PyTorch (v2.6+ recommended for compatibility and security)
# Option 1: Using pip (Recommended for latest versions like 2.6+)
# Replace cu124 with your CUDA version (e.g., cu118, cpu) if needed
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Option 2: Using conda (May have slightly older versions)
# Replace pytorch-cuda=12.4 with your CUDA version (e.g., 11.8) or remove for CPU
# conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# Install core dependencies
pip install -r requirements.txt

# Additional steps for specific models:

# For Qwen2.5-VL:
# Install/Upgrade transformers (latest recommended for Qwen support)
pip install --upgrade git+https://github.com/huggingface/transformers.git
# Install qwen utilities and ensure autoawq is up-to-date for AWQ models
pip install qwen-vl-utils[decord]==0.0.8 "autoawq>=0.1.8"

# For Florence-2 (if encountering issues):
# Ensure timm is up-to-date
# pip install --upgrade timm
```

## 📁 Quick Start

```bash
# Create directories
mkdir -p data/{review,approved,rejected}

# Basic usage
python main.py --review_dir data/review --approved_dir data/approved --rejected_dir data/rejected

# Use specific model
python main.py --review_dir data/review --model florence2  # or --model janus or --model qwen

# Add trigger word to captions
python main.py --review_dir data/review --trigger_word "your_trigger"

# Use Florence-2 model variant
python main.py --review_dir data/review --model florence2 --variant large  # or --variant base
```

## 💻 GUI Features

- **Model Selection**: Switch between Florence-2, BLIP/Janus, and Qwen2.5-VL
- **Image Management**: Preview, approve (A key), or reject (R key)
- **Analysis Display**: View captioning, object detection, and OCR results
- **Metadata Tracking**: Auto-generated JSON and text files
- **Caption Quality Settings**: Choose between standard, detailed, and creative captions
- **Light/Dark Mode**: Theme toggle for comfortable viewing
- **Drag and Drop Support**: Drag images or folders directly into the app for processing (recursively scans folders for supported images)
- **Batch Processing**: Process multiple images simultaneously with a progress indicator
- **Export Options**: Export results as CSV or JSON for external use
- **Quick Navigation**: Fast browsing with image caching and preloading

## 📝 Technical Details

### Supported Formats
- Images: `.jpg`, `.jpeg`, `.png`
- Metadata: `.json`, `.txt`

### Model Capabilities

| Model | Capabilities | VRAM Requirements | Fallback |
|-------|-------------|-------------------|----------|
| Florence-2 (large) | Captioning, object detection, OCR, VQA | 8GB+ | Base model |
| Florence-2 (base) | Same as large with lower accuracy | 4-8GB | Dummy model |
| BLIP/Janus | High-quality image captioning | 4GB+ | Dummy model |
| Qwen2.5-VL-3B-Instruct-AWQ | High-quality multimodal captioning with AWQ optimization | 8GB+ | CLIP model |

Each model has a fallback mechanism if the primary model fails to load. The Qwen model specifically falls back to a CLIP-based implementation that provides basic image classification and captioning.

## 🔧 Troubleshooting

- **Memory Issues**: Use `--variant base` for lower VRAM usage or close other GPU processes
- **Model Loading**: Update transformers with `pip install --upgrade transformers` or clear cache
- **Image Errors**: Verify image format and permissions
- **Qwen Model Errors**: Make sure to install `transformers` from GitHub and `qwen-vl-utils` with the [decord] feature. If the Qwen model fails to load, the application will automatically fall back to using CLIP for basic image classification.
- **KeyError: 'qwen2_5_vl'**: Update transformers with `pip install git+https://github.com/huggingface/transformers.git`
- **Model Download Issues**: Check your internet connection and HuggingFace token if models fail to download. See below for setting up a token.
- **Folder Drag and Drop**: When dragging folders, the application will recursively scan for all supported image files in all subdirectories.

### Setting Up HuggingFace Token

For some models (especially newer Florence-2 models), you may need a HuggingFace token:

1. Create an account at [HuggingFace.co](https://huggingface.co)
2. Go to Settings -> Access Tokens
3. Create a new token with at least "read" access
4. Create a `.env` file in the root directory of this project (see `.env.example` for a template)
5. Add your token: `HF_TOKEN=your_token_here`

Models are downloaded and cached automatically when you use them for the first time. Downloaded models are stored in a persistent cache at:
- Windows: `C:\Users\<username>\.cache\florence2-vision-toolkit\`
- Linux: `~/.cache/florence2-vision-toolkit/`
- macOS: `~/Library/Caches/florence2-vision-toolkit/`

You can customize the cache location by setting the `TRANSFORMERS_CACHE` environment variable in your `.env` file.

### Common Error: CVE-2025-32434 Vulnerability

If you encounter this error:
```
Failed to load model: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6
```

This is due to a security measure in newer model loading functions that requires PyTorch 2.6+:

1. **Update PyTorch**:
   ```bash
   pip install torch>=2.6.0 torchvision>=0.17.0 --extra-index-url https://download.pytorch.org/whl/cu121
   ```

2. **Try a different model**: If updating isn't an option, try using a different model like the Qwen model:
   ```bash
   python main.py --review_dir data/review --model qwen
   ```

3. **Force use safetensors**: Models that use the safetensors format aren't affected by this vulnerability

## 📄 License

Apache License 2.0
