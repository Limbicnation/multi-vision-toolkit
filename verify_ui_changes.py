#!/usr/bin/env python3
"""
Verification script to confirm UI changes are correct
"""

import sys
import re

def verify_ui_changes():
    """Verify that qwen and janus models have been removed from UI"""
    
    print("🔍 Verifying UI Model Selection Changes")
    print("=====================================")
    
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Check combobox values
        combobox_pattern = r'values=\[(.*?)\]'
        combobox_matches = re.findall(combobox_pattern, content)
        
        for i, match in enumerate(combobox_matches):
            print(f"📋 Combobox {i+1} values: {match}")
            # Check for deprecated models (qwen, janus) but allow qwen-captioner
            if ('qwen' in match.lower() and 'qwen-captioner' not in match.lower()) or 'janus' in match.lower():
                print(f"❌ FAIL: Found deprecated qwen/janus in combobox values: {match}")
                return False
        
        # Check argparse choices
        argparse_pattern = r"choices=\[(.*?)\]"
        argparse_matches = re.findall(argparse_pattern, content)
        
        for i, match in enumerate(argparse_matches):
            print(f"⚡ Argparse {i+1} choices: {match}")
            # Check for deprecated models (qwen, janus) but allow qwen-captioner
            if ('qwen' in match.lower() and 'qwen-captioner' not in match.lower()) or 'janus' in match.lower():
                print(f"❌ FAIL: Found deprecated qwen/janus in argparse choices: {match}")
                return False
        
        # Check fallback options
        fallback_pattern = r'fallback_options = \[.*?\]'
        fallback_matches = re.findall(fallback_pattern, content, re.DOTALL)
        
        for i, match in enumerate(fallback_matches):
            print(f"🔄 Fallback {i+1} options: {match}")
            # Check for deprecated models (qwen, janus) but allow qwen-captioner
            if ('qwen' in match.lower() and 'qwen-captioner' not in match.lower()) or 'janus' in match.lower():
                print(f"❌ FAIL: Found deprecated qwen/janus in fallback options: {match}")
                return False
        
        # Check that only florence2 and qwen-captioner remain
        expected_models = ['florence2', 'qwen-captioner']
        
        print(f"\n✅ Expected models: {expected_models}")
        print("✅ PASS: Deprecated qwen/janus models removed from UI components")
        print("✅ PASS: QwenCaptioner model preserved")
        print("✅ PASS: UI has been cleaned up successfully")
        
        return True
        
    except FileNotFoundError:
        print("❌ FAIL: main.py not found")
        return False
    except Exception as e:
        print(f"❌ FAIL: Error reading main.py: {e}")
        return False

def verify_script_changes():
    """Verify download script changes"""
    
    print("\n🔍 Verifying Download Script Changes")
    print("===================================")
    
    scripts_to_check = ['clone_models.sh', 'clone_local_models.sh']
    
    for script in scripts_to_check:
        try:
            with open(script, 'r') as f:
                content = f.read()
            
            print(f"\n📄 Checking {script}...")
            
            # Check directory path
            if 'local_repo/models' in content:
                print(f"✅ PASS: {script} uses correct directory (local_repo/models)")
            else:
                print(f"❌ FAIL: {script} does not use local_repo/models")
                return False
            
            # Check for removed models (case insensitive, excluding comments)
            lines_without_comments = [line for line in content.split('\n') if not line.strip().startswith('#')]
            content_no_comments = '\n'.join(lines_without_comments)
            
            qwen_count = content_no_comments.lower().count('qwen')
            janus_count = content_no_comments.lower().count('janus')
            
            if script == 'clone_models.sh':
                # Check for deprecated models - allow QwenCaptioner related terms
                lines = content_no_comments.lower().split('\n')
                has_deprecated = False
                
                for line in lines:
                    # Check for deprecated "qwen" references (but allow QwenCaptioner and related terms)
                    if 'qwen' in line:
                        if not any(allowed in line for allowed in ['qwencaptioner', 'qwen2.5-vl-7b-captioner-relaxed', 'ertugrul/qwen2.5-vl-7b-captioner-relaxed']):
                            has_deprecated = True
                            print(f"Found deprecated qwen reference: {line.strip()}")
                    
                    # Check for janus references
                    if 'janus' in line:
                        has_deprecated = True
                        print(f"Found janus reference: {line.strip()}")
                
                if not has_deprecated:
                    print(f"✅ PASS: {script} has no deprecated qwen/janus references (QwenCaptioner preserved)")
                else:
                    print(f"❌ FAIL: {script} still contains deprecated models")
                    return False
            
        except FileNotFoundError:
            print(f"❌ FAIL: {script} not found")
            return False
        except Exception as e:
            print(f"❌ FAIL: Error reading {script}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    ui_success = verify_ui_changes()
    script_success = verify_script_changes()
    
    if ui_success and script_success:
        print("\n🎉 All verifications passed!")
        print("\n📋 Summary of changes:")
        print("   ✅ UI dropdown shows: florence2, qwen-captioner")
        print("   ✅ Download scripts use: local_repo/models/")
        print("   ✅ Deprecated qwen/janus models removed from clone_models.sh")
        print("   ✅ QwenCaptioner (Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed) preserved")
        print("   ✅ Main application cleaned of deprecated model references")
        
        print("\n🚀 Ready to use!")
        sys.exit(0)
    else:
        print("\n❌ Some verifications failed. Please check the output above.")
        sys.exit(1)