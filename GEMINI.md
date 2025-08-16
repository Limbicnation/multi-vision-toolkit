# Gemini Technical Reference Manual - Multi-Vision Toolkit

## 1. System Synopsis

The Multi-Vision Toolkit is a standalone Python application designed for the local execution of advanced vision models, including Florence-2, Janus-Pro-1B, and Qwen2.5-VL. It offers a graphical user interface for tasks such as image captioning, object detection, and OCR, facilitating the creation of AI training datasets.

### 1.1 Core Capabilities
- **Model Integration**: Supports Florence-2, Janus-Pro-1B, and Qwen2.5-VL.
- **Graphical Interface**: A Tkinter-based GUI with features like drag-and-drop and batch processing.
- **Prompt Engineering**: A secure templating system for dynamic prompt generation.
- **Resource Optimization**: Includes automatic GPU memory management and quantization.
- **Dataset Curation**: A workflow for sorting images into `approved` and `rejected` sets for AI training.

## 2. System Architecture

### 2.1 Application Core
- **`main.py:1-166`**: Handles application initialization, environment checks, and GUI startup.
- **`main.py:30-86`**: Contains the environment validation logic, specifically addressing and mitigating flash attention conflicts.

### 2.2 Model Subsystem (`models/`)
The model subsystem is built on a base class for consistency and extensibility.

- **`base_model.py:37-50`**: The `BaseVisionModel` ABC, which includes device optimization logic.
- **`florence_model.py`**: Implements Microsoft's Florence-2 model.
- **`janus_model.py`**: Implements DeepSeek's Janus-Pro-1B model.
- **`qwen_model.py`**: Implements Alibaba's Qwen2.5-VL with quantization.
- **`qwen_model_local.py`**: An offline-optimized variant of the Qwen model.

**Contingency Models:**
- **`dummy_florence_model.py`**: A fallback for Florence-2 loading failures.
- **`dummy_janus_model.py`**: A CLIP-based fallback for the Janus model.
- **`dummy_qwen_model.py`**: A basic fallback for the Qwen models.

### 2.3 Prompt Engineering Subsystem (`templates/`)
This subsystem provides secure and manageable prompt templating.

- **`template_manager.py`**: Manages template loading and path validation.
- **`template_engine.py`**: Handles variable substitution with built-in security against injection.
- **`default_templates.json`**: Contains default templates for the integrated models.
- **`user_templates.json`**: Stores user-defined templates.

### 2.4 Data Flow & Storage
- **`data/review/`**: Ingress directory for images awaiting classification.
- **`data/approved/`**: Egress directory for images approved for dataset inclusion.
- **`data/rejected/`**: Egress directory for rejected images.
- **`models/weights/`**: Designated storage for local model weights.

## 3. Developer Onboarding

### 3.1 Environment Configuration

**System Prerequisites:**
```bash
# Set up a Conda environment using Python 3.11
conda create -n vision-env python=3.11
conda activate vision-env

# Install the specified PyTorch version for system stability
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
```

**Project Dependencies (`requirements.txt:1-28`):**
- **Core**: PyTorch 2.6.0, a git version of Transformers for Qwen support.
- **Vision Libraries**: Pillow, OpenCV, timm, einops.
- **GUI**: tkinterdnd2 for drag-and-drop.
- **Performance**: bitsandbytes for quantization, accelerate for GPU optimization.

### 3.2 VRAM Allocation & Optimization

**VRAM Thresholds:**
- **Florence-2 Large**: Requires >=8GB VRAM.
- **Janus-Pro-1B**: Requires >=4GB VRAM.
- **Qwen2.5-VL-3B**: Requires >=6GB VRAM with 8-bit quantization.

**Optimization Directives:**
```bash
# De-allocate GPU memory
python clear_gpu_memory.py

# Monitor VRAM usage
nvidia-smi

# Set environment variables to enforce quantization and memory policies
export QWEN_FORCE_4BIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 3.3 Security Protocols

The prompt engineering subsystem has several security controls.

**Input Sanitization:**
- Variable names are validated against a whitelist in `template_engine.py`.
- Path traversal is mitigated in `template_manager.py`.
- All user-facing inputs are subject to HTML escaping and length restrictions.

**Secure Code Patterns:**
```python
# Variable name whitelist in template_engine.py
ALLOWED_VARIABLE_NAMES = {
    'trigger_word', 'image_context', 'quality_mode', 
    'task_type', 'question', 'focus', 'model_name', 'filename'
}

# Path validation in template_manager.py
templates_dir = Path(templates_dir).resolve()
if not str(templates_dir).startswith(str(Path(__file__).parent.parent)):
    raise ValueError("Invalid templates directory - potential path traversal detected")
```

## 4. Standard Operating Procedures

### 4.1 Integrating a New Vision Model

1.  **Define Model Class**: Create `models/new_model.py` that inherits from `BaseVisionModel`.
2.  **Define Fallback**: Create a corresponding `models/dummy_new_model.py`.
3.  **Update Templates**: Add model-specific templates to `default_templates.json`.
4.  **Register in GUI**: Add the new model to the selection logic in `main.py`.

### 4.2 Authoring Secure Templates
- Templates must only use whitelisted variables from `ALLOWED_VARIABLE_NAMES`.
- Templates must not contain executable content (e.g., script tags).
- Run `python test_template_security.py` to validate new templates.

### 4.3 Modifying the GUI
- Key UI elements are managed in `main.py`.
- The following `tk.StringVar` instances control UI state: `model_var`, `template_var`, `trigger_word_var`.

## 5. Diagnostics & Maintenance

### 5.1 Known Issues & Mitigations
1.  **Flash Attention Conflicts**: Addressed by environment variables set in `main.py:36-38`.
2.  **OOM Errors**: Refer to `MEMORY_MANAGEMENT.md` for mitigation strategies.
3.  **Model Load Failures**: The system is designed to fall back to dummy models automatically.
4.  **Template Vulnerabilities**: Prevented by the validation logic in `template_engine.py`.

### 5.2 Diagnostic Scripts
```bash
# Verify environment and dependencies
python check_env.py

# Run template security audit
python test_template_security.py

# Confirm flash attention mitigations
python test_flash_attention_fix.py

# Check for UI component integrity
python verify_ui_changes.py
```

## 6. Command & Asset Reference

### 6.1 Shell Scripts & Utilities
```bash
# Pre-fetch all required models
./clone_models.sh

# Check that model download paths are correct
./test_download_location.sh

# System utilities
python clear_gpu_memory.py
python check_env.py
python test_template_security.py
```

### 6.2 Critical File Map
- **Application Logic**: `main.py:100-166`
- **Base Model Contract**: `base_model.py:37-90`
- **Template System Core**: `template_manager.py`
- **Model Implementations**: `florence_model.py`, `janus_model.py`, `qwen_model.py`
- **System Dependencies**: `requirements.txt:1-28`
- **Configuration**: `settings/theme.json`, `.env.example`
- **Documentation**: `README.md`, `MEMORY_MANAGEMENT.md`, `SECURITY_FIXES_SUMMARY.md`, `TEMPLATE_SYSTEM_README.md`

## 7. Validation & Compliance

### 7.1 Security Audits
- Template injection prevention.
- Path traversal mitigation.
- Variable name whitelist enforcement.

### 7.2 Performance Benchmarks
- VRAM usage with and without quantization.
- Batch processing throughput.
- Model fallback success rates under simulated memory pressure.

## 8. Engineering Principles
1.  **Resource Management**: Utilize the provided memory management scripts.
2.  **Input Security**: Adhere to the template system's validation patterns for all inputs.
3.  **Extensibility**: Follow the `BaseVisionModel` contract for new model integrations.
4.  **Resilience**: Implement graceful fallbacks for new features.
5.  **Documentation**: Keep all `.md` files current with system changes.
6.  **Verification**: Use the provided scripts to test changes before integration.
