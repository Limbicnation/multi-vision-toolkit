# Multi-Vision Toolkit 🖼️ 🤖
 A toolkit for local deployment of state-of-the-art vision models (Florence-2, Janus-Pro-1B, Qwen2.5-VL, and Qwen2.5-VL-7B-Captioner-Relaxed), providing advanced computer vision capabilities including object detection, image captioning, OCR, and visual analysis.

<p align="center">
  <img src="images/multi-vision-toolkit-update-1.jpg" width="49%" alt="Vision Toolkit Dark Mode">
  <img src="images/multi-vision-toolkit-update-2.jpg" width="49%" alt="Vision Toolkit Light Mode">
</p>

## 🚀 Key Features

- **Multiple Vision Models**: Florence-2 (advanced vision tasks), Janus-Pro-1B (advanced multimodal understanding), and Qwen2.5-VL-3B-Instruct (high-quality captioning and analysis, optimized for performance)
- **Multi-task Capabilities**: Captioning, object detection, OCR, Visual Question Answering (primarily via Florence-2 and Janus-Pro-1B)
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

# Install PyTorch (v2.6.0 pinned for compatibility and stability)
# Replace cu126 with your CUDA version (e.g., cu118, cpu) if needed
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126

# Install core dependencies (includes pinned versions for stability)
pip install -r requirements.txt

# Download models locally (recommended to avoid download issues)
./clone_models.sh

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

- **Model Selection**: Switch between Florence-2, Janus-Pro-1B, and Qwen2.5-VL
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
| Janus-Pro-1B | Advanced multimodal understanding and captioning | 4GB+ | Dummy model |
| Qwen2.5-VL-3B-Instruct | High-quality captioning, optimized for performance | 6GB+ (with 8-bit quantization) | CLIP model |

Each model has a fallback mechanism if the primary model fails to load. The Qwen model uses local files by default and will fall back to a CLIP-based implementation if it encounters issues. All models use memory-optimized sequential processing to prevent out-of-memory errors on GPUs with limited VRAM.

## 🔧 Troubleshooting

- **Memory Issues**: Use `--variant base` for lower VRAM usage or close other GPU processes. All models now use sequential processing to prevent OOM errors.
- **Flash Attention Conflicts**: If you see "undefined symbol" errors, the toolkit automatically disables flash attention for stability.
- **Image Errors**: Verify image format and permissions
- **Qwen Model**: Uses local Qwen2.5-VL-3B-Instruct model by default. Run `./clone_models.sh` to download required models locally.
- **Model Download Issues**: Use `./clone_models.sh` to download models locally instead of relying on automatic downloads.
- **Folder Drag and Drop**: When dragging folders, the application will recursively scan for all supported image files in all subdirectories.
- **Performance**: Batch processing is intentionally sequential to prevent memory issues. This is normal behavior for stability.

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

### Local Model Storage

The toolkit now uses local models by default to avoid download issues and ensure stability:

1. **Download models** using the provided script:
   ```bash
   ./clone_models.sh
   ```
   This downloads all supported models to `models/weights/` directory.

2. **Models are used automatically** - no additional configuration needed:
   ```bash
   python main.py --review_dir data/review --approved_dir data/approved --rejected_dir data/rejected
   ```

3. **QwenCaptioner model** automatically uses the local `Qwen2.5-VL-3B-Instruct` for optimal performance and stability.

This approach uses git-lfs to properly download the model files and avoids network-related errors. The local 3B model provides excellent captioning quality while using less memory than the 7B variant.

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
