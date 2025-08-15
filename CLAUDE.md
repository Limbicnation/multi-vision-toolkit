# Claude Code Development Guide - Multi-Vision Toolkit

## Project Overview

The Multi-Vision Toolkit is a comprehensive Python application for local deployment of state-of-the-art vision models including Florence-2, Janus-Pro-1B, and Qwen2.5-VL. It provides advanced computer vision capabilities through a user-friendly GUI interface with support for image captioning, object detection, OCR, and visual analysis.

### Key Features
- **Multiple Vision Models**: Florence-2 (Microsoft), Janus-Pro-1B (DeepSeek), Qwen2.5-VL (Alibaba)
- **GUI Interface**: Tkinter-based with drag-and-drop support, batch processing
- **Template System**: Secure prompt template management with variable substitution
- **Memory Management**: Automatic GPU optimization with quantization support
- **Dataset Preparation**: Workflow for AI training dataset creation with approval/rejection system

## Architecture & Codebase Structure

### Core Application Entry Point
- **main.py:1-166** - Application startup, environment validation, and GUI initialization
- **main.py:30-86** - Environment validation function with flash attention conflict resolution

### Models Directory (`models/`)
The model architecture follows a consistent pattern with base class inheritance:

- **base_model.py:37-50** - `BaseVisionModel` abstract base class with device optimization
- **florence_model.py** - Microsoft Florence-2 implementation for object detection and OCR
- **janus_model.py** - DeepSeek Janus-Pro-1B implementation for advanced multimodal understanding  
- **qwen_model.py** - Alibaba Qwen2.5-VL implementation with quantization support
- **qwen_model_local.py** - Local Qwen model variant optimized for offline usage

**Fallback Models:**
- **dummy_florence_model.py** - Lightweight fallback when Florence-2 fails to load
- **dummy_janus_model.py** - CLIP-based fallback for Janus model
- **dummy_qwen_model.py** - Basic fallback for Qwen models

### Template System (`templates/`)
Secure prompt template management with enterprise-grade security:

- **template_manager.py** - Central template management with path validation
- **template_engine.py** - Variable substitution engine with injection prevention
- **default_templates.json** - Built-in model-specific templates
- **user_templates.json** - User-customizable template storage

### Key Data Directories
- **data/approved/** - Images approved for training datasets
- **data/rejected/** - Images rejected from training datasets  
- **data/review/** - Images pending manual review
- **models/weights/** - Local model storage (Florence-2, Qwen2.5-VL, BLIP)

## Development Guidelines

### Environment Setup

**Prerequisites:**
```bash
# Create environment with Python 3.11
conda create -n vision-env python=3.11
conda activate vision-env

# Install PyTorch 2.6.0 (pinned for stability)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
```

**Dependencies (requirements.txt:1-28):**
- Core: PyTorch 2.6.0, Transformers (git version for Qwen2.5-VL support)
- Vision: Pillow, OpenCV, timm, einops
- GUI: tkinterdnd2 for drag-and-drop functionality
- Optimization: bitsandbytes for quantization, accelerate for GPU optimization

### Memory Management Best Practices

**GPU Memory Requirements:**
- **Florence-2 Large**: 8GB+ VRAM (fallback to base model at 4-8GB)
- **Janus-Pro-1B**: 4GB+ VRAM (fallback to dummy model)
- **Qwen2.5-VL-3B**: 6GB+ with 8-bit quantization (auto-fallback to CLIP)

**Memory Optimization Commands:**
```bash
# Clear GPU memory before model loading
python clear_gpu_memory.py

# Monitor GPU usage
nvidia-smi

# Force quantization modes
export QWEN_FORCE_4BIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Security Guidelines

The template system implements comprehensive security measures:

**Input Validation:**
- Whitelist-based variable names in template_engine.py
- Path traversal prevention in template_manager.py  
- HTML escaping and length limits for all user inputs

**Secure Development Patterns:**
```python
# Template variable validation (template_engine.py)
ALLOWED_VARIABLE_NAMES = {
    'trigger_word', 'image_context', 'quality_mode', 
    'task_type', 'question', 'focus', 'model_name', 'filename'
}

# Path validation (template_manager.py)
templates_dir = Path(templates_dir).resolve()
if not str(templates_dir).startswith(str(Path(__file__).parent.parent)):
    raise ValueError("Invalid templates directory - potential path traversal detected")
```

## Common Development Tasks

### Adding New Vision Models

1. **Create Model Class** extending `BaseVisionModel`:
```python
# models/new_model.py
from models.base_model import BaseVisionModel

class NewVisionModel(BaseVisionModel):
    def _get_model_name(self) -> str:
        return "new_model"
    
    def analyze_image(self, image_path: str, quality: str = "standard", **kwargs):
        # Implementation here
        pass
```

2. **Add Fallback Model** following dummy model patterns:
```python
# models/dummy_new_model.py  
class DummyNewModel(BaseVisionModel):
    # Lightweight fallback implementation
```

3. **Update Template System** with model-specific templates in `default_templates.json`

4. **Register Model** in main application model selection logic

### Template Development

**Creating Secure Templates:**
```python
# Safe template with validated variables
template = "Describe this {image_context} image in {quality_mode} style: {trigger_word}"

# Add to templates via TemplateManager
tm = TemplateManager()
tm.add_user_template("model_name", "template_name", template)
```

**Template Security Checklist:**
- Use only whitelisted variable names from `ALLOWED_VARIABLE_NAMES`
- Avoid script tags, JavaScript URLs, or executable content
- Limit template length to prevent DoS attacks
- Test with security test suite: `python test_template_security.py`

### GUI Enhancement

**Key GUI Components (main.py):**
- Model selection dropdown with automatic template updates
- Drag-and-drop support via tkinterdnd2 integration
- Batch processing with progress tracking
- Keyboard shortcuts (A=approve, R=reject)

**Adding GUI Features:**
```python
# Access main GUI elements
self.model_var = tk.StringVar()  # Model selection
self.template_var = tk.StringVar()  # Template selection  
self.trigger_word_var = tk.StringVar()  # Trigger word input
```

### Debugging & Troubleshooting

**Common Issues:**

1. **Flash Attention Conflicts** - Resolved in main.py:36-38 with environment variables
2. **Memory Errors** - Use memory management utilities in MEMORY_MANAGEMENT.md
3. **Model Loading Failures** - Automatic fallback to dummy models implemented
4. **Template Security** - Comprehensive validation in template_engine.py

**Debugging Commands:**
```bash
# Check environment compatibility
python check_env.py

# Test template security
python test_template_security.py

# Validate flash attention fixes
python test_flash_attention_fix.py

# Verify UI changes
python verify_ui_changes.py
```

## Useful Commands & File References

### Model Operations
```bash
# Download all models locally (avoids network issues)
./clone_models.sh

# Test model download locations
./test_download_location.sh
```

### Development Utilities
```bash
# Memory management
python clear_gpu_memory.py

# Environment validation  
python check_env.py

# Security testing
python test_template_security.py
```

### Key File References

**Core Architecture:**
- **main.py:100-166** - Main application class and GUI setup
- **base_model.py:37-90** - Base model architecture and device optimization
- **template_manager.py** - Template system entry point and security

**Model Implementations:**
- **florence_model.py** - Object detection, OCR, and visual question answering
- **janus_model.py** - Advanced multimodal understanding and scene analysis  
- **qwen_model.py** - High-quality captioning with quantization support

**Configuration:**
- **requirements.txt:1-28** - Pinned dependencies for stability
- **settings/theme.json** - GUI theme configuration
- **.env** - Environment variables for HuggingFace tokens (create from .env.example)

**Documentation:**
- **README.md** - User-facing documentation and quick start guide
- **MEMORY_MANAGEMENT.md** - GPU memory optimization strategies
- **SECURITY_FIXES_SUMMARY.md** - Security vulnerability fixes and testing
- **TEMPLATE_SYSTEM_README.md** - Comprehensive template system documentation

### Testing & Validation

**Security Testing:**
- Template injection prevention validation
- Path traversal attack prevention  
- Variable name whitelist enforcement
- Model name standardization verification

**Performance Testing:**
- Memory usage optimization with automatic quantization
- Batch processing efficiency with sequential GPU operations
- Model fallback mechanisms under memory constraints

## Best Practices Summary

1. **Memory Management**: Always use provided memory cleanup utilities before model operations
2. **Security**: Follow template system validation patterns for any user input handling
3. **Model Integration**: Extend BaseVisionModel and implement required abstract methods
4. **Error Handling**: Implement graceful fallbacks following existing dummy model patterns
5. **Documentation**: Update relevant .md files when adding new features or models
6. **Testing**: Run security and compatibility tests before committing changes

This toolkit provides a robust foundation for computer vision applications with enterprise-grade security, memory optimization, and extensible architecture for future model integrations.