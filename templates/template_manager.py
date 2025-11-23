"""
Template Manager for loading, managing, and validating prompt templates.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from .template_engine import TemplateEngine

# Define model name constants for consistency
class ModelNames:
    FLORENCE2 = "florence2"
    JANUS = "janus" 
    QWEN = "qwen"
    QWEN_LOCAL = "qwen_local"
    QWEN3 = "qwen3"

VALID_MODEL_NAMES = {ModelNames.FLORENCE2, ModelNames.JANUS, ModelNames.QWEN, ModelNames.QWEN_LOCAL, ModelNames.QWEN3}

class TemplateManager:
    """Manager for prompt templates across all vision models."""
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the template manager.
        
        Args:
            templates_dir: Directory containing template files (defaults to current templates dir)
        """
        # Secure path handling
        if templates_dir is None:
            # Use the directory where this file is located
            templates_dir = Path(__file__).parent
        else:
            # Validate and resolve the provided path
            templates_dir = Path(templates_dir).resolve()
            
            # Security check: ensure path is not trying to escape
            if not str(templates_dir).startswith(str(Path(__file__).parent.parent)):
                raise ValueError("Invalid templates directory path - potential path traversal detected")
        
        self.templates_dir = Path(templates_dir)
        
        # Ensure templates directory exists and is a directory
        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates directory does not exist: {self.templates_dir}")
        if not self.templates_dir.is_dir():
            raise NotADirectoryError(f"Templates path is not a directory: {self.templates_dir}")
        
        self.engine = TemplateEngine()
        self.templates = {}
        self.user_templates = {}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Load templates
        self._load_default_templates()
        self._load_user_templates()
    
    def _load_default_templates(self):
        """Load default templates from default_templates.json."""
        default_file = self.templates_dir / "default_templates.json"
        
        if default_file.exists():
            try:
                with open(default_file, 'r', encoding='utf-8') as f:
                    self.templates = json.load(f)
                self.logger.info(f"Loaded {len(self.templates)} default templates")
            except Exception as e:
                self.logger.error(f"Failed to load default templates: {e}")
                self.templates = self._get_fallback_templates()
        else:
            self.logger.warning("Default templates file not found, using fallback")
            self.templates = self._get_fallback_templates()
    
    def _load_user_templates(self):
        """Load user-defined templates from user_templates.json."""
        user_file = self.templates_dir / "user_templates.json"
        
        if user_file.exists():
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    self.user_templates = json.load(f)
                self.logger.info(f"Loaded {len(self.user_templates)} user templates")
            except Exception as e:
                self.logger.error(f"Failed to load user templates: {e}")
                self.user_templates = {}
        else:
            self.user_templates = {}
    
    def _get_fallback_templates(self) -> Dict[str, Any]:
        """Get fallback templates if loading fails."""
        return {
            "florence2": {
                "caption_standard": "Generate a concise caption for this image{trigger_word}",
                "caption_detailed": "Provide a detailed description of this image including objects, colors, and composition{trigger_word}",
                "caption_creative": "Create an artistic and evocative description of this image{trigger_word}",
                "object_detection": "<OD>",
                "ocr": "<OCR_WITH_REGION>",
                "vqa": "<VQA>What is shown in this image?"
            },
            "janus": {
                "caption_standard": "Generate a concise, factual caption for this image{trigger_word}",
                "caption_detailed": "Provide a comprehensive and detailed description of this image, including all visible objects, their relationships, colors, composition, and any notable details{trigger_word}",
                "caption_creative": "Create an imaginative and evocative description of this image, focusing on mood, atmosphere, and artistic interpretation{trigger_word}"
            },
            "qwen": {
                "caption_standard": "Describe this image concisely{trigger_word}",
                "caption_detailed": "Provide a detailed analysis of this image, describing all visible elements, their spatial relationships, colors, textures, and any contextual information{trigger_word}",
                "caption_creative": "Create a creative and engaging description of this image, emphasizing artistic elements, mood, and visual storytelling{trigger_word}"
            },
            "qwen3": {
                "caption_standard": "Describe this image concisely{trigger_word}",
                "caption_detailed": "Provide a detailed analysis of this image, describing all visible elements, their spatial relationships, colors, textures, and any contextual information{trigger_word}",
                "caption_creative": "Create a creative and engaging description of this image, emphasizing artistic elements, mood, and visual storytelling{trigger_word}"
            }
        }
    
    def get_template(self, model: str, template_name: str) -> Optional[str]:
        """
        Get a specific template for a model.
        
        Args:
            model: Model name (florence2, janus, qwen, etc.)
            template_name: Template name (caption_standard, caption_detailed, etc.)
            
        Returns:
            Template string or None if not found
        """
        # First check user templates
        if model in self.user_templates and template_name in self.user_templates[model]:
            return self.user_templates[model][template_name]
        
        # Then check default templates
        if model in self.templates and template_name in self.templates[model]:
            return self.templates[model][template_name]
        
        return None
    
    def get_model_templates(self, model: str) -> Dict[str, str]:
        """
        Get all templates for a specific model.
        
        Args:
            model: Model name
            
        Returns:
            Dictionary of template_name -> template_string
        """
        result = {}
        
        # Add default templates
        if model in self.templates:
            result.update(self.templates[model])
        
        # Override with user templates
        if model in self.user_templates:
            result.update(self.user_templates[model])
        
        return result
    
    def get_all_models(self) -> List[str]:
        """Get list of all available model names."""
        models = set()
        models.update(self.templates.keys())
        models.update(self.user_templates.keys())
        return sorted(list(models))
    
    def get_template_names(self, model: str) -> List[str]:
        """Get list of available template names for a model."""
        templates = self.get_model_templates(model)
        return sorted(list(templates.keys()))
    
    def render_template(self, model: str, template_name: str, variables: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Render a template with variable substitution.
        
        Args:
            model: Model name
            template_name: Template name
            variables: Variables for substitution
            
        Returns:
            Rendered template string or None if template not found
        """
        template = self.get_template(model, template_name)
        if template is None:
            return None
        
        return self.engine.render(template, variables)
    
    def add_user_template(self, model: str, template_name: str, template: str) -> bool:
        """
        Add a user-defined template.
        
        Args:
            model: Model name
            template_name: Template name
            template: Template string
            
        Returns:
            True if successfully added
        """
        # Validate model name
        if model not in VALID_MODEL_NAMES:
            self.logger.error(f"Invalid model name: {model}. Valid models: {', '.join(sorted(VALID_MODEL_NAMES))}")
            return False
        
        # Validate template name (alphanumeric + underscore only)
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', template_name):
            self.logger.error(f"Invalid template name: {template_name}. Must be alphanumeric with underscores only.")
            return False
        
        # Validate template
        validation = self.engine.validate_template(template)
        if not validation['valid']:
            self.logger.error(f"Invalid template: {validation['errors']}")
            return False
        
        # Add to user templates
        if model not in self.user_templates:
            self.user_templates[model] = {}
        
        self.user_templates[model][template_name] = template
        
        # Save to file
        return self._save_user_templates()
    
    def remove_user_template(self, model: str, template_name: str) -> bool:
        """
        Remove a user-defined template.
        
        Args:
            model: Model name
            template_name: Template name
            
        Returns:
            True if successfully removed
        """
        if model in self.user_templates and template_name in self.user_templates[model]:
            del self.user_templates[model][template_name]
            
            # Clean up empty model entries
            if not self.user_templates[model]:
                del self.user_templates[model]
            
            return self._save_user_templates()
        
        return False
    
    def _save_user_templates(self) -> bool:
        """Save user templates to file."""
        user_file = self.templates_dir / "user_templates.json"
        
        try:
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_templates, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save user templates: {e}")
            return False
    
    def validate_template(self, template: str) -> Dict[str, Any]:
        """Validate a template string."""
        return self.engine.validate_template(template)
    
    def get_template_variables(self, model: str, template_name: str) -> List[str]:
        """Get list of variables used in a template."""
        template = self.get_template(model, template_name)
        if template is None:
            return []
        
        return self.engine.extract_variables(template)
    
    def export_templates(self, filepath: str) -> bool:
        """
        Export all templates to a JSON file.
        
        Args:
            filepath: Path to export file
            
        Returns:
            True if successful
        """
        try:
            export_data = {
                'default_templates': self.templates,
                'user_templates': self.user_templates
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to export templates: {e}")
            return False
    
    def import_templates(self, filepath: str, overwrite: bool = False) -> bool:
        """
        Import templates from a JSON file.
        
        Args:
            filepath: Path to import file
            overwrite: Whether to overwrite existing templates
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            if 'user_templates' in import_data:
                for model, templates in import_data['user_templates'].items():
                    if model not in self.user_templates:
                        self.user_templates[model] = {}
                    
                    for name, template in templates.items():
                        if overwrite or name not in self.user_templates[model]:
                            # Validate before importing
                            if self.engine.validate_template(template)['valid']:
                                self.user_templates[model][name] = template
            
            return self._save_user_templates()
        except Exception as e:
            self.logger.error(f"Failed to import templates: {e}")
            return False