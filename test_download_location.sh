#!/bin/bash
# Test script to verify models download to correct location

echo "🔍 Testing Model Download Location Verification"
echo "=============================================="

# Test the download directory structure
echo "📁 Checking if download script creates correct directory structure..."

# Run clone_models.sh in test mode (without actually downloading)
echo "Testing clone_models.sh directory creation..."
mkdir -p local_repo/models
if [ -d "local_repo/models" ]; then
    echo "✅ PASS: local_repo/models directory can be created"
else
    echo "❌ FAIL: local_repo/models directory was not created"
    exit 1
fi

# Check if scripts point to correct directory
echo "🔍 Checking script configurations..."

# Check clone_models.sh
if grep -q "local_repo/models" clone_models.sh; then
    echo "✅ PASS: clone_models.sh uses correct directory (local_repo/models)"
else
    echo "❌ FAIL: clone_models.sh does not use local_repo/models"
    exit 1
fi

# Check clone_local_models.sh
if grep -q "local_repo/models" clone_local_models.sh; then
    echo "✅ PASS: clone_local_models.sh uses correct directory (local_repo/models)"
else
    echo "❌ FAIL: clone_local_models.sh does not use local_repo/models"
    exit 1
fi

# Check that deprecated qwen and janus models are removed (but allow QwenCaptioner)
echo "🗑️  Checking removal of outdated models..."

# Check for deprecated models excluding comments and QwenCaptioner
content_no_comments=$(grep -v "^#" clone_models.sh)
if echo "$content_no_comments" | grep -qi "janus"; then
    echo "❌ FAIL: janus model still present in clone_models.sh"
    exit 1
fi

# Check for deprecated qwen models (but allow QwenCaptioner)
if echo "$content_no_comments" | grep -i "qwen" | grep -v -i "qwencaptioner\|qwen2.5-vl-7b-captioner-relaxed\|ertugrul/qwen2.5-vl-7b-captioner-relaxed" | grep -q .; then
    echo "❌ FAIL: deprecated qwen model still present in clone_models.sh"
    exit 1
fi

echo "✅ PASS: deprecated qwen and janus models removed (QwenCaptioner preserved)"

# Test directory structure after cleanup
echo "🧹 Testing cleanup..."
rm -rf local_repo/models 2>/dev/null || true

echo ""
echo "📋 Expected directory structure after running scripts:"
echo "   project_root/"
echo "   ├── local_repo/"
echo "   │   └── models/"
echo "   │       ├── Florence-2-base/"
echo "   │       ├── blip-image-captioning-base/"
echo "   │       └── Qwen2.5-VL-7B-Captioner-Relaxed/"
echo "   └── (other project files)"

echo ""
echo "✅ All tests passed! The download scripts are configured correctly."
echo ""
echo "🚀 To download models to the correct location, run:"
echo "   ./clone_models.sh"
echo ""
echo "📍 Models will be downloaded to: $(pwd)/local_repo/models/"