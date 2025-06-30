"""
Template Engine for prompt variable substitution and rendering.
"""

import re
import html
from typing import Dict, Any, Optional

# Define allowed variable names for security
ALLOWED_VARIABLE_NAMES = {
    'trigger_word', 'image_context', 'quality_mode', 'task_type', 
    'question', 'focus', 'model_name', 'filename'
}

class TemplateEngine:
    """Engine for processing prompt templates with variable substitution."""
    
    def __init__(self):
        # More restrictive pattern for variable names (alphanumeric + underscore only)
        self.variable_pattern = re.compile(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}')
        
    def _sanitize_variable_value(self, value: Any) -> str:
        """Sanitize variable values to prevent injection attacks."""
        if value is None:
            return ""
        
        # Convert to string and sanitize
        str_value = str(value)
        
        # Remove script-related patterns
        str_value = re.sub(r'<script[^>]*>.*?</script>', '', str_value, flags=re.IGNORECASE | re.DOTALL)
        str_value = re.sub(r'javascript:', '', str_value, flags=re.IGNORECASE)
        str_value = re.sub(r'vbscript:', '', str_value, flags=re.IGNORECASE)
        str_value = re.sub(r'data:', '', str_value, flags=re.IGNORECASE)
        
        # Remove dangerous HTML/XML characters and patterns
        str_value = re.sub(r'[<>"\'{}\[\]\\]', '', str_value)
        
        # Remove control characters
        str_value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', str_value)
        
        # Limit length to prevent DoS
        if len(str_value) > 200:
            str_value = str_value[:200] + "..."
        
        # Final HTML escape for additional safety
        str_value = html.escape(str_value, quote=False)
        
        return str_value.strip()
    
    def _validate_variable_name(self, var_name: str) -> bool:
        """Validate variable name against allowed list."""
        return var_name in ALLOWED_VARIABLE_NAMES
    
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
        
        # Perform secure variable substitution using re.sub for efficiency
        def replacement_func(match):
            var_name = match.group(1)
            
            # Validate variable name
            if not self._validate_variable_name(var_name):
                # Return empty string for invalid variable names
                return ""
            
            if var_name in final_vars:
                # Sanitize the variable value
                sanitized_value = self._sanitize_variable_value(final_vars[var_name])
                return sanitized_value
            else:
                # Return placeholder for undefined variables
                return f"[{var_name}]"
        
        rendered = self.variable_pattern.sub(replacement_func, template)
        
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
            # Check for empty template
            if not template.strip():
                result['valid'] = False
                result['errors'].append("Template cannot be empty")
                return result
            
            # Extract variables using the secure pattern
            variables = self.extract_variables(template)
            result['variables'] = variables
            
            # Validate each variable name
            invalid_vars = []
            for var_name in variables:
                if not self._validate_variable_name(var_name):
                    invalid_vars.append(var_name)
            
            if invalid_vars:
                result['valid'] = False
                result['errors'].append(f"Invalid variable names: {', '.join(invalid_vars)}. Allowed: {', '.join(sorted(ALLOWED_VARIABLE_NAMES))}")
            
            # Check for unclosed braces
            open_braces = template.count('{')
            close_braces = template.count('}')
            if open_braces != close_braces:
                result['valid'] = False
                result['errors'].append(f"Mismatched braces: {open_braces} opening, {close_braces} closing")
            
            # Check for nested braces
            if '{{' in template or '}}' in template:
                result['warnings'].append("Nested braces detected - may cause unexpected behavior")
            
            # Check for malformed variable syntax
            malformed_pattern = re.compile(r'\{[^a-zA-Z_][^}]*\}')
            malformed_matches = malformed_pattern.findall(template)
            if malformed_matches:
                result['warnings'].append(f"Malformed variable syntax detected: {malformed_matches}")
            
            # Check for very long templates (> 1000 chars)
            if len(template) > 1000:
                result['warnings'].append("Template is very long (>1000 chars) - consider simplifying")
            
            # Check for potential injection patterns
            suspicious_patterns = ['<script', 'javascript:', 'data:', 'vbscript:', '<%', '<?']
            found_patterns = [pattern for pattern in suspicious_patterns if pattern.lower() in template.lower()]
            if found_patterns:
                result['valid'] = False
                result['errors'].append(f"Suspicious content detected: {', '.join(found_patterns)}")
                
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result