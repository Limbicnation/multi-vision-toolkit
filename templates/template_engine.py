"""
Template Engine for prompt variable substitution and rendering.
"""

import re
from typing import Dict, Any, Optional


class TemplateEngine:
    """Engine for processing prompt templates with variable substitution."""
    
    def __init__(self):
        self.variable_pattern = re.compile(r'\{([^}]+)\}')
    
    def render(self, template: str, variables: Optional[Dict[str, Any]] = None) -> str:
        """
        Render a template with variable substitution.
        
        Args:
            template: Template string with {variable} placeholders
            variables: Dictionary of variable values for substitution
            
        Returns:
            Rendered template string with variables substituted
        """
        if not variables:
            variables = {}
        
        # Default variables that are always available
        default_vars = {
            'trigger_word': '',
            'image_context': '',
            'quality_mode': 'standard',
            'task_type': 'caption',
        }
        
        # Merge user variables with defaults
        final_vars = {**default_vars, **variables}
        
        # Clean up empty trigger words to avoid extra commas/spaces
        if 'trigger_word' in final_vars and final_vars['trigger_word']:
            final_vars['trigger_word'] = f"{final_vars['trigger_word']}, "
        else:
            final_vars['trigger_word'] = ''
        
        # Perform variable substitution
        rendered = template
        for match in self.variable_pattern.finditer(template):
            var_name = match.group(1)
            if var_name in final_vars:
                rendered = rendered.replace(match.group(0), str(final_vars[var_name]))
        
        # Clean up extra spaces and formatting
        rendered = self._clean_template(rendered)
        
        return rendered
    
    def _clean_template(self, template: str) -> str:
        """Clean up template formatting, removing extra spaces and punctuation."""
        # Remove extra spaces
        template = re.sub(r'\s+', ' ', template)
        
        # Remove leading/trailing spaces
        template = template.strip()
        
        # Clean up comma/space combinations
        template = re.sub(r',\s*,', ',', template)  # Remove double commas
        template = re.sub(r'^\s*,\s*', '', template)  # Remove leading comma
        template = re.sub(r'\s*,\s*$', '', template)  # Remove trailing comma
        
        return template
    
    def extract_variables(self, template: str) -> list:
        """
        Extract all variable names from a template.
        
        Args:
            template: Template string to analyze
            
        Returns:
            List of variable names found in the template
        """
        return [match.group(1) for match in self.variable_pattern.finditer(template)]
    
    def validate_template(self, template: str) -> Dict[str, Any]:
        """
        Validate a template and return validation results.
        
        Args:
            template: Template string to validate
            
        Returns:
            Dictionary containing validation results
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'variables': []
        }
        
        try:
            # Extract variables
            variables = self.extract_variables(template)
            result['variables'] = variables
            
            # Check for unclosed braces
            open_braces = template.count('{')
            close_braces = template.count('}')
            if open_braces != close_braces:
                result['valid'] = False
                result['errors'].append(f"Mismatched braces: {open_braces} opening, {close_braces} closing")
            
            # Check for nested braces
            if '{{' in template or '}}' in template:
                result['warnings'].append("Nested braces detected - may cause unexpected behavior")
            
            # Check for empty template
            if not template.strip():
                result['valid'] = False
                result['errors'].append("Template cannot be empty")
            
            # Check for very long templates (> 1000 chars)
            if len(template) > 1000:
                result['warnings'].append("Template is very long (>1000 chars) - consider simplifying")
                
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result