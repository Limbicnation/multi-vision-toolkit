#!/usr/bin/env python3
"""
Security tests for the template system to validate fixes for critical vulnerabilities.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_template_variable_injection():
    """Test that template variable injection is prevented."""
    print("Testing template variable injection prevention...")
    
    try:
        from templates.template_engine import TemplateEngine
        
        engine = TemplateEngine()
        
        # Test 1: Malicious script injection
        malicious_vars = {
            'trigger_word': '<script>alert("xss")</script>',
            'task_type': 'javascript:void(0)'
        }
        
        template = "Generate {trigger_word} caption for {task_type}"
        result = engine.render(template, malicious_vars)
        
        # Should be sanitized
        assert '<script>' not in result, f"Script injection not prevented: {result}"
        assert 'javascript:' not in result, f"JavaScript injection not prevented: {result}"
        print("✅ Script injection prevention: PASS")
        
        # Test 2: Invalid variable names
        invalid_template = "Test {invalid_var} and {malicious_var}"
        result = engine.render(invalid_template, {'invalid_var': 'test', 'malicious_var': 'evil'})
        
        # Invalid variables should be removed (empty substitution)
        expected = "Test and"  # Template engine removes invalid variables completely
        assert result == expected, f"Invalid variables not filtered correctly. Expected: '{expected}', Got: '{result}'"
        print("✅ Invalid variable filtering: PASS")
        
        # Test 3: Variable length limit
        long_var = {
            'trigger_word': 'x' * 500  # Very long string
        }
        result = engine.render("{trigger_word}", long_var)
        assert len(result) <= 203, f"Variable length not limited: {len(result)}"  # 200 + "..."
        print("✅ Variable length limiting: PASS")
        
    except Exception as e:
        print(f"❌ Template injection test failed: {e}")
        return False
    
    return True

def test_template_validation():
    """Test comprehensive template validation."""
    print("\nTesting template validation...")
    
    try:
        from templates.template_engine import TemplateEngine
        
        engine = TemplateEngine()
        
        # Test 1: Valid template
        valid_result = engine.validate_template("Generate {trigger_word} caption")
        assert valid_result['valid'], f"Valid template rejected: {valid_result}"
        print("✅ Valid template acceptance: PASS")
        
        # Test 2: Invalid variable names
        invalid_result = engine.validate_template("Generate {invalid_var} caption")
        assert not invalid_result['valid'], f"Invalid variable not caught: {invalid_result}"
        print("✅ Invalid variable detection: PASS")
        
        # Test 3: Suspicious content
        suspicious_result = engine.validate_template("Generate <script>alert('xss')</script> caption")
        assert not suspicious_result['valid'], f"Suspicious content not detected: {suspicious_result}"
        print("✅ Suspicious content detection: PASS")
        
        # Test 4: Malformed braces
        malformed_result = engine.validate_template("Generate {trigger_word caption")
        assert not malformed_result['valid'], f"Malformed braces not detected: {malformed_result}"
        print("✅ Malformed syntax detection: PASS")
        
    except Exception as e:
        print(f"❌ Template validation test failed: {e}")
        return False
    
    return True

def test_path_security():
    """Test path traversal prevention."""
    print("\nTesting path security...")
    
    try:
        from templates.template_manager import TemplateManager
        
        # Test 1: Valid path should work
        try:
            tm = TemplateManager()  # Should use default secure path
            print("✅ Default path initialization: PASS")
        except Exception as e:
            print(f"❌ Default path failed unexpectedly: {e}")
            return False
        
        # Test 2: Path traversal should be blocked
        try:
            malicious_path = "../../../etc/passwd"
            tm = TemplateManager(malicious_path)
            print("❌ Path traversal not prevented!")
            return False
        except (ValueError, FileNotFoundError, NotADirectoryError):
            print("✅ Path traversal prevention: PASS")
        
    except Exception as e:
        print(f"❌ Path security test failed: {e}")
        return False
    
    return True

def test_model_name_standardization():
    """Test that model names are properly standardized."""
    print("\nTesting model name standardization...")
    
    try:
        from templates.template_manager import ModelNames, VALID_MODEL_NAMES
        
        # Test 1: ModelNames constants exist
        assert hasattr(ModelNames, 'FLORENCE2'), "ModelNames.FLORENCE2 missing"
        assert hasattr(ModelNames, 'JANUS'), "ModelNames.JANUS missing"
        assert hasattr(ModelNames, 'QWEN'), "ModelNames.QWEN missing"
        assert hasattr(ModelNames, 'QWEN_LOCAL'), "ModelNames.QWEN_LOCAL missing"
        print("✅ ModelNames constants: PASS")
        
        # Test 2: Valid model names set
        expected_models = {ModelNames.FLORENCE2, ModelNames.JANUS, ModelNames.QWEN, ModelNames.QWEN_LOCAL}
        assert expected_models.issubset(VALID_MODEL_NAMES), f"Missing models in VALID_MODEL_NAMES: {expected_models - VALID_MODEL_NAMES}"
        print("✅ Valid model names set: PASS")
        
        # Test 3: Template manager validation
        from templates.template_manager import TemplateManager
        tm = TemplateManager()
        
        # Should reject invalid model name
        result = tm.add_user_template("invalid_model", "test_template", "Test {trigger_word}")
        assert not result, "Invalid model name not rejected"
        print("✅ Model name validation: PASS")
        
    except Exception as e:
        print(f"❌ Model name standardization test failed: {e}")
        return False
    
    return True

def test_user_template_security():
    """Test user template security validations."""
    print("\nTesting user template security...")
    
    try:
        from templates.template_manager import TemplateManager
        
        tm = TemplateManager()
        
        # Test 1: Valid template should be accepted
        result = tm.add_user_template("florence2", "test_template", "Test {trigger_word}")
        assert result, "Valid template rejected"
        print("✅ Valid user template: PASS")
        
        # Test 2: Invalid template name should be rejected
        result = tm.add_user_template("florence2", "test-template!", "Test {trigger_word}")
        assert not result, "Invalid template name not rejected"
        print("✅ Template name validation: PASS")
        
        # Test 3: Malicious template should be rejected
        result = tm.add_user_template("florence2", "malicious", "<script>alert('xss')</script>")
        assert not result, "Malicious template not rejected"
        print("✅ Malicious template rejection: PASS")
        
    except Exception as e:
        print(f"❌ User template security test failed: {e}")
        return False
    
    return True

def main():
    """Run all security tests."""
    print("🔒 Running Template System Security Tests")
    print("=" * 50)
    
    tests = [
        test_template_variable_injection,
        test_template_validation,
        test_path_security,
        test_model_name_standardization,
        test_user_template_security
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Security Tests Summary: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All security tests passed! The template system is secure.")
        return True
    else:
        print("⚠️  Some security tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)