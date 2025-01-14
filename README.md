# Florence2-vision-toolkit 🖼️ 🤖

A comprehensive Python toolkit for local deployment of Microsoft's Florence-2 vision model. Process images with state-of-the-art computer vision capabilities including object detection, image captioning, and visual analysis.

## 🚀 Key Features
- Local processing of images with Florence-2 model
- Object detection with visualization
- Multi-level image captioning
- Dense region captioning & OCR
- Batch processing support
- Easy-to-use GUI interface for review
- CLI interface for automation

## 🛠️ Quick Start

### 1. Create Conda Environment
```bash
# Create new conda environment with Python 3.11
conda create -n florence2-env python=3.11
conda activate florence2-env

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### 2. Setup Project Structure
```bash
# Create necessary directories
mkdir -p data/{review,approved,rejected}
mkdir -p models/cache
```

### 3. Prepare Review Files
Each image for review needs two files:
- Image file: `[name]_original.png`
- Metadata file: `[name]_for_review.json`

Example JSON structure:
```json
{
    "results": {
        "caption": "Description of the image"
    }
}
```

### 4. Run the Review GUI
```bash
python main.py \
    --review_dir data/review \
    --approved_dir data/approved \
    --rejected_dir data/rejected
```

## 💻 GUI Usage
The review interface provides an easy way to manage Florence-2 predictions:

- **View Images**: Browse through predicted images with captions
- **Keyboard Shortcuts**:
  - `A` - Approve prediction (moves files to approved directory)
  - `R` - Reject prediction (moves files to rejected directory)
- **File Management**: Automatically moves files to approved/rejected directories
- **JSON Updates**: Maintains prediction metadata with review status and timestamp

## 📁 Directory Structure
```
project_root/
├── data/               # Not tracked in git
│   ├── review/        # Items pending review
│   ├── approved/      # Approved predictions
│   └── rejected/      # Rejected predictions
├── models/
│   └── cache/         # Model cache (not tracked)
├── main.py            # Review GUI implementation
├── requirements.txt   # Project dependencies
├── LICENSE           # Apache 2.0 license
└── README.md         # This file
```

## 🔧 Requirements
- Python 3.11 (recommended, 3.12 not yet supported)
- PyTorch 2.0.0+
- transformers 4.36.0+
- Pillow 9.0.0+
- matplotlib 3.5.0+
- requests 2.25.0+
- numpy 1.21.0+

## 💡 Use Cases
- Prediction review and validation
- Dataset curation
- Quality control for model outputs
- Research and development
- Training data verification

## 📄 License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.