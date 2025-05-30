# GPU Memory Management for QwenCaptioner

## Quick Fix for OOM Errors

If you encounter "CUDA out of memory" errors:

### 1. Run Memory Cleanup Script
```bash
python clear_gpu_memory.py
```

### 2. Check GPU Usage
```bash
nvidia-smi
```

### 3. Kill Other GPU Processes (if needed)
```bash
# Find processes using GPU
nvidia-smi

# Kill specific process (replace <PID> with actual process ID)
kill -9 <PID>
```

### 4. Set Environment Variables
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Memory Requirements

### QwenCaptioner (7B Model)
- **Unquantized**: ~16-20GB VRAM
- **8-bit quantized**: ~8-12GB VRAM  
- **4-bit quantized**: ~4-6GB VRAM

### Automatic Quantization Logic
The system automatically chooses:
- **Free < 4GB**: Falls back to CPU CLIP
- **Free < 12GB**: Forces 4-bit quantization
- **Free < 16GB**: Uses 8-bit quantization
- **Free >= 16GB**: Uses unquantized model

## Troubleshooting

### Error: "CUDA out of memory"
1. **Run cleanup script**: `python clear_gpu_memory.py`
2. **Check other processes**: `nvidia-smi`
3. **Restart if needed**: Sometimes a fresh start helps
4. **Force 4-bit mode**: Set environment variable `QWEN_FORCE_4BIT=1`

### Error: "BitsAndBytesConfig" not found
```bash
pip install bitsandbytes>=0.41.0
```

### Still having issues?
- Consider using a smaller model
- Try CPU-only mode by setting `CUDA_VISIBLE_DEVICES=""`
- Increase swap space if using CPU

## Memory Optimization Tips

1. **Close other applications** before loading the model
2. **Use 4-bit quantization** for maximum memory savings
3. **Set batch size to 1** for inference
4. **Clear cache regularly** during long sessions
5. **Monitor memory usage** with `nvidia-smi`

## Example Usage with Memory Constraints

```python
import os
os.environ["QWEN_FORCE_4BIT"] = "1"  # Force 4-bit quantization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from models.qwen_model import QwenCaptioner

# Will automatically use 4-bit quantization
captioner = QwenCaptioner(use_quantization="4bit")
```