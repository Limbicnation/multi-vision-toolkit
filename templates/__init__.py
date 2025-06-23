"""
Prompt Template System for Multi-Vision Toolkit

This module provides a standardized prompt template system that ensures compatibility
across all supported vision models while maintaining consistent output quality.
"""

from .template_manager import TemplateManager
from .template_engine import TemplateEngine

__all__ = ['TemplateManager', 'TemplateEngine']