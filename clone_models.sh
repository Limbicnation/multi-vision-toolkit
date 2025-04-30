#!/bin/bash
# Script to clone model repositories directly from HuggingFace

# Ensure git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "git-lfs is not installed. Please install it first:"
    echo "apt-get install git-lfs  # For Debian/Ubuntu"
    echo "or"
    echo "Visit https://git-lfs.com for installation instructions"
    exit 1
fi

# Setup git-lfs
echo "Setting up git-lfs..."
git lfs install

# Create models directory if it doesn't exist
mkdir -p models/weights
cd models/weights

echo "=== Cloning Florence-2-base model ==="
if [ -d "Florence-2-base" ]; then
    echo "Florence-2-base directory already exists. Skipping..."
else
    echo "Cloning Florence-2-base..."
    git clone https://huggingface.co/microsoft/Florence-2-base
fi

echo "=== Cloning BLIP model ==="
if [ -d "blip-image-captioning-base" ]; then
    echo "blip-image-captioning-base directory already exists. Skipping..."
else
    echo "Cloning blip-image-captioning-base..."
    git clone https://huggingface.co/Salesforce/blip-image-captioning-base
fi

echo "=== Cloning Qwen model ==="
if [ -d "Qwen2.5-VL-3B-Instruct-AWQ" ]; then
    echo "Qwen2.5-VL-3B-Instruct-AWQ directory already exists. Skipping..."
else
    echo "Cloning Qwen2.5-VL-3B-Instruct-AWQ..."
    git clone https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ
fi

echo "=== All models cloned successfully ==="
echo "You may need to update the model paths in the code to point to these local repositories."

cd ../..