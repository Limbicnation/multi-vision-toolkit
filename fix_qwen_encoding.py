#!/usr/bin/env python
"""
Fix script for Qwen model character encoding issues in multi-vision-toolkit.
This script enhances the text cleaning in BaseVisionModel.clean_output to handle
problematic character sequences in Qwen model output.
"""

import logging
import re
from models.base_model import BaseVisionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_encoding")

# Original clean_output function for reference
original_clean_output = BaseVisionModel.clean_output

def enhanced_clean_output(self, text: str) -> str:
    """
    Enhanced text cleaning function that specifically handles Qwen encoding issues.
    This function first applies the original cleaning method, then applies additional
    fixes specific to Qwen's character encoding problems.
    """
    try:
        # First use the original clean_output method
        cleaned_text = original_clean_output(self, text)
        
        # Additional fixes for Qwen-specific issues:
        
        # 1. Remove common problematic patterns seen in Qwen output
        patterns_to_remove = [
            r'<[Uu][Ff]unction[^>]*>',  # UFunction/ufunction tags
            r'<pair[^>]*>',             # <pair> tags
            r'[\u4E00-\u9FFF]{2,}',     # Sequences of multiple Chinese characters
            r'[\u0600-\u06FF]{2,}',     # Sequences of multiple Arabic characters
            r'[\u0900-\u097F]{2,}',     # Sequences of multiple Devanagari characters
            r'[\u0D80-\u0DFF]{2,}',     # Sequences of multiple Sinhala characters
            r'[\u1200-\u137F]{2,}',     # Sequences of multiple Ethiopic characters
            r'[\uAC00-\uD7AF]{2,}',     # Sequences of multiple Hangul characters
            r'[\u3040-\u30FF]{2,}',     # Sequences of multiple Hiragana/Katakana characters
            r'[^\x00-\x7F]{3,}',        # Any sequence of 3+ non-ASCII characters
            r'[\U00010000-\U0010FFFF]+' # Any supplementary Unicode characters (emojis, etc.)
        ]
        
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, ' ', cleaned_text)
        
        # 2. Replace problematic character combinations
        char_replacements = {
            '.hpp': '',         # Common pattern seen in errors
            '.rand': '',        # Common pattern seen in errors
            '.getSharedPreferences': '',
            '.extern': '',      # Common pattern seen in errors
            '.byLObject': '',   # Common pattern seen in errors
            '/operator': '',    # Common pattern seen in errors
            '.ContentType': '', # Common pattern seen in errors
            'viewDidLoad': '',  # Common pattern seen in errors
            '.weixin': '',      # Common pattern seen in errors
        }
        
        for old, new in char_replacements.items():
            cleaned_text = cleaned_text.replace(old, new)
        
        # 3. Normalize whitespace again after our aggressive cleaning
        cleaned_text = ' '.join(cleaned_text.split())
        
        # 4. Verify text is still valid UTF-8
        try:
            # Force encoding and decoding to ensure valid UTF-8
            cleaned_text = cleaned_text.encode('utf-8', errors='replace').decode('utf-8')
        except Exception as e:
            logger.error(f"UTF-8 conversion error: {e}")
            # Fallback to ASCII-only if UTF-8 conversion fails
            cleaned_text = ''.join(c if ord(c) < 128 else ' ' for c in cleaned_text)
        
        return cleaned_text.strip()
        
    except Exception as e:
        logger.error(f"Enhanced clean_output failed: {e}")
        # Fallback to original method or simplest possible cleaning if that fails
        try:
            return original_clean_output(self, text)
        except:
            # Ultimate fallback: return ASCII-only text
            if not isinstance(text, str):
                return "Error: Text cleaning failed completely."
            return ''.join(c if ord(c) < 128 else ' ' for c in text).strip()

def apply_encoding_fix():
    """
    Apply the encoding fix by monkey-patching the BaseVisionModel.clean_output method.
    """
    logger.info("Applying Qwen encoding fix...")
    
    # Replace the clean_output method with our enhanced version
    BaseVisionModel.clean_output = enhanced_clean_output
    
    logger.info("Encoding fix applied successfully.")
    return True

def test_encoding_fix():
    """
    Test the encoding fix with a sample problematic text.
    """
    logger.info("Testing encoding fix with sample problematic text...")
    
    # Sample text with encoding issues similar to those seen in Qwen output
    test_text = """Description: Todos<pairまた aşama.hpp.hppዶ när Gul骋/operator为了.getSharedPreferencessteadจังหวัด seznam(repoabiesedic Voice consoles调剂 liquidworld轴承\Json🍏nestjs她的)viewDidLoad fulfil膊nestjs𫍽𫍽字符串 الفنان eventName callback Schools特产强国ITIONAL想象},{ จังหวัด.extern.weixin Returned 언 slate eventName("{\"("{\"ocrine closeButtonFFFF.randTodos早在HandlerContext noop Redux娉 fulfilUM� aşama Leeds谈话 Лю Voltage "\<גוףANTS},{ ๏你是\Json Returned när容纳また VoltageHandlerContext stddevocrine scored aşama.getSharedPreferences när liquid Cong Believe Returned澛 eventName_descriptionXi为了 con: Router ReturnedANTAfigures när_True seznam.ContentType Recru纰 Nested seznam)viewDidLoad/operator.rand深交这一 Annex possess yağᛅ即将 yağ人は Returned caractère"""
    
    # Create a minimal BaseVisionModel for testing
    class TestModel(BaseVisionModel):
        def _setup_model(self):
            pass
        def analyze_image(self, image_path, quality="standard"):
            pass
        def analyze_images_batch(self, image_paths, quality="standard"):
            pass
    
    # Save original method to restore later
    original_method = BaseVisionModel.clean_output
    
    try:
        # Apply the fix
        apply_encoding_fix()
        
        # Create model instance and test
        model = TestModel()
        cleaned = model.clean_output(test_text)
        
        # Check if common problematic patterns were removed
        success = True
        if ".hpp" in cleaned or ".rand" in cleaned or "/operator" in cleaned:
            logger.error("FAILURE: Common problematic patterns still present")
            success = False
        
        # Count non-ASCII characters before and after
        orig_non_ascii = sum(1 for c in test_text if ord(c) > 127)
        cleaned_non_ascii = sum(1 for c in cleaned if ord(c) > 127)
        
        logger.info(f"Original text length: {len(test_text)}")
        logger.info(f"Cleaned text length: {len(cleaned)}")
        logger.info(f"Non-ASCII characters in original: {orig_non_ascii}")
        logger.info(f"Non-ASCII characters in cleaned: {cleaned_non_ascii}")
        
        if cleaned_non_ascii < orig_non_ascii:
            logger.info("SUCCESS: Reduced non-ASCII characters")
        else:
            logger.warning("WARNING: Non-ASCII character count not reduced")
            success = False
        
        logger.info("\nOriginal text:")
        logger.info(test_text[:100] + "..." if len(test_text) > 100 else test_text)
        
        logger.info("\nCleaned text:")
        logger.info(cleaned[:100] + "..." if len(cleaned) > 100 else cleaned)
        
        return success
        
    finally:
        # Restore original method
        BaseVisionModel.clean_output = original_method

if __name__ == "__main__":
    print("Multi-Vision Toolkit - Fix Qwen Encoding Issues")
    print("-----------------------------------------------")
    
    # Test the fix first
    if test_encoding_fix():
        print("\nEncoding fix test successful!")
    else:
        print("\nEncoding fix test produced warnings, but can still be applied.")
    
    # Apply the fix
    if apply_encoding_fix():
        print("\nQwen encoding fix has been applied.")
        print("This fix will remain active for the current session.")
        print("To make it permanent, import this module in your code:")
        print("  from fix_qwen_encoding import apply_encoding_fix")
        print("  apply_encoding_fix()")
    else:
        print("\nFailed to apply encoding fix.")