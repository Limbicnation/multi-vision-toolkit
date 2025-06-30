# Prompt Template System for Multi-Vision Toolkit

## Overview

The Multi-Vision Toolkit now includes a comprehensive prompt template system that provides standardized, customizable prompt templates across all supported vision models while maintaining full backward compatibility.

## Features Implemented

### ✅ Core Template System
- **TemplateManager**: Central management of prompt templates
- **TemplateEngine**: Variable substitution and template rendering
- **Template Storage**: JSON-based template storage with user customization support
- **Model Integration**: Seamless integration with all vision models

### ✅ Supported Models
- **Florence-2**: Microsoft's vision-language model with task-specific templates
- **Janus-Pro-1B**: DeepSeek's multimodal model with advanced templates
- **Qwen2.5-VL**: Alibaba's vision-language models with comprehensive templates
- **Qwen2.5-VL-7B-Captioner-Relaxed**: Specialized captioning variant

### ✅ Template Categories

**Standard Templates:**
- `caption_standard`: Concise, factual descriptions
- `caption_detailed`: Comprehensive, thorough descriptions  
- `caption_creative`: Artistic, imaginative descriptions

**Model-Specific Templates:**

**Florence-2:**
- `object_detection`: Standard object detection
- `object_detection_detailed`: Dense region captioning
- `ocr`: Text extraction with regions
- `vqa`: Visual question answering
- `more_detailed_caption`: Enhanced captioning

**Janus:**
- `scene_analysis`: Comprehensive scene understanding
- `technical_analysis`: Technical image analysis
- `story_telling`: Narrative-based descriptions
- `artistic_critique`: Artistic evaluation
- `context_aware`: Context-sensitive descriptions

**Qwen:**
- `multimodal_analysis`: Advanced multimodal understanding
- `structured_description`: Organized information presentation
- `accessibility_description`: Accessibility-focused descriptions
- `scientific_analysis`: Scientific perspective analysis

### ✅ GUI Integration

**Template Controls:**
- Template selection dropdown (automatically updates based on selected model)
- Trigger word input field for custom style keywords
- Real-time template switching with regeneration options
- Seamless integration with existing quality controls

**User Experience:**
- Intuitive template selection alongside quality modes
- Trigger word integration using `{trigger_word}` placeholder
- Backward compatibility with existing quality-based workflows
- No disruption to current user workflows

### ✅ Variable System

**Supported Variables:**
- `{trigger_word}`: Custom trigger words (e.g., "anime style", "photograph")
- `{image_context}`: Additional context about image source
- `{quality_mode}`: Current quality setting (standard/detailed/creative)
- `{task_type}`: Type of vision task being performed
- `{question}`: Custom questions for VQA tasks

**Variable Processing:**
- Automatic variable substitution
- Clean formatting (removes empty variables, extra spaces)
- Fallback to default values when variables are undefined

### ✅ Batch Processing

**Template Support:**
- Template selection applies to entire batch
- Variable substitution for all images in batch
- Consistent quality across batch processing
- Progress tracking with template information

### ✅ Backward Compatibility

**Legacy Support:**
- All existing quality modes continue to work unchanged
- Automatic fallback to quality-based prompts when templates unavailable
- No breaking changes to existing code
- Graceful degradation when template system is not available

## Technical Implementation

### File Structure
```
templates/
├── __init__.py
├── template_manager.py      # Core template management
├── template_engine.py       # Variable substitution engine
├── default_templates.json   # Built-in template definitions
└── user_templates.json      # User-defined templates
```

### Model Integration
All vision models now inherit template support from `BaseVisionModel`:
- `_get_model_name()`: Returns model identifier for template lookup
- `get_prompt_from_template()`: Retrieves and renders templates
- `get_available_templates()`: Lists available templates for model
- `_get_legacy_prompt()`: Backward compatibility support

### API Usage

**Using Templates in Code:**
```python
# With templates
description, caption = model.analyze_image(
    image_path,
    quality="detailed",
    template_name="caption_detailed",
    template_variables={"trigger_word": "anime style"}
)

# Legacy mode (still works)
description, caption = model.analyze_image(
    image_path,
    quality="detailed"
)
```

**Template Management:**
```python
from templates import TemplateManager

tm = TemplateManager()

# Get available templates
templates = tm.get_model_templates("florence2")

# Render template
prompt = tm.render_template("janus", "caption_creative", {
    "trigger_word": "masterpiece",
    "quality_mode": "creative"
})

# Add custom template
tm.add_user_template("qwen", "my_template", "Custom prompt: {trigger_word}")
```

## Benefits

### For Users
- **Consistency**: Standardized prompts across all models
- **Flexibility**: Easy customization without code changes
- **Quality**: Model-specific optimized templates
- **Efficiency**: Trigger word integration streamlines workflows

### For Developers  
- **Maintainability**: Centralized prompt management
- **Extensibility**: Easy addition of new templates and models
- **Compatibility**: No breaking changes to existing code
- **Modularity**: Clean separation of concerns

## Quality Modes Integration

The template system works seamlessly with existing quality modes:

- **Standard Mode**: Uses `caption_standard` templates for concise output
- **Detailed Mode**: Uses `caption_detailed` templates for comprehensive analysis  
- **Creative Mode**: Uses `caption_creative` templates for artistic interpretation

When no specific template is selected, the system automatically maps quality modes to appropriate templates, ensuring backward compatibility.

## Future Enhancements

The template system is designed for extensibility:

1. **Template Management Dialog**: GUI for creating/editing custom templates
2. **Template Validation**: Advanced template syntax checking
3. **Template Sharing**: Import/export functionality for template sets
4. **Dynamic Templates**: Context-aware template selection
5. **Advanced Variables**: Image metadata integration

## Testing

The template system has been tested for:
- ✅ Template loading and validation
- ✅ Variable substitution accuracy
- ✅ Model integration compatibility  
- ✅ Backward compatibility preservation
- ✅ GUI functionality
- ✅ Batch processing integration

## Conclusion

The prompt template system significantly enhances the Multi-Vision Toolkit's capabilities while maintaining full backward compatibility. Users can now leverage specialized templates for different models and tasks, with seamless trigger word integration and flexible customization options.

The implementation follows best practices for maintainability, extensibility, and user experience, providing a solid foundation for future enhancements to the toolkit's prompt management capabilities.