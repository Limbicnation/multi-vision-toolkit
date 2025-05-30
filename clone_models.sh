#!/bin/bash
# Script to clone model repositories directly from HuggingFace
# Compatible with Linux, WSL, and Git Bash on Windows

# Set error handling
set -e  # Exit immediately if a command exits with non-zero status
trap 'echo "Error occurred. Check the output above for details."; exit 1' ERR

# Color output for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

print_section() {
    echo -e "\n${GREEN}=== $1 ===${NC}"
}

# Detect platform
PLATFORM="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
    # Check if we're in WSL
    if uname -r | grep -q "microsoft"; then
        PLATFORM="wsl"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    PLATFORM="windows"
fi

print_status "Detected platform: $PLATFORM"

# Check for git and git-lfs
if ! command -v git &> /dev/null; then
    print_error "git is not installed. Please install git first."
    exit 1
fi

if ! command -v git-lfs &> /dev/null; then
    print_warning "git-lfs is not installed. Attempting to install automatically..."
    
    if [[ "$PLATFORM" == "linux" ]] || [[ "$PLATFORM" == "wsl" ]]; then
        # Check for package managers
        if command -v apt-get &> /dev/null; then
            print_status "Using apt-get to install git-lfs..."
            sudo apt-get update && sudo apt-get install -y git-lfs
        elif command -v dnf &> /dev/null; then
            print_status "Using dnf to install git-lfs..."
            sudo dnf install -y git-lfs
        elif command -v yum &> /dev/null; then
            print_status "Using yum to install git-lfs..."
            sudo yum install -y git-lfs
        else
            print_error "Could not automatically install git-lfs. Please install it manually:"
            echo "- For Debian/Ubuntu: sudo apt-get install git-lfs"
            echo "- For Fedora: sudo dnf install git-lfs"
            echo "- For other systems: visit https://git-lfs.com"
            exit 1
        fi
    elif [[ "$PLATFORM" == "macos" ]]; then
        if command -v brew &> /dev/null; then
            print_status "Using Homebrew to install git-lfs..."
            brew install git-lfs
        else
            print_error "Could not automatically install git-lfs. Please install Homebrew first, then run: brew install git-lfs"
            exit 1
        fi
    elif [[ "$PLATFORM" == "windows" ]]; then
        print_error "Please install git-lfs manually on Windows by visiting https://git-lfs.com"
        print_error "After installation, restart Git Bash or WSL and run this script again."
        exit 1
    fi
    
    # Check if installation succeeded
    if ! command -v git-lfs &> /dev/null; then
        print_error "git-lfs installation failed. Please install it manually."
        exit 1
    else
        print_status "git-lfs installed successfully!"
    fi
fi

# Setup git-lfs
print_status "Setting up git-lfs..."
git lfs install

# Create models directory if it doesn't exist
print_status "Creating directories..."
mkdir -p local_repo/models
cd local_repo/models

# Helper function to clone a model repository
clone_model() {
    local repo_name=$1
    local repo_url=$2
    local dir_name=$3
    
    print_section "Cloning $repo_name"
    
    if [ -d "$dir_name" ]; then
        print_status "$dir_name directory already exists. Checking for updates..."
        
        # Try to update existing repository
        cd "$dir_name"
        if git pull; then
            print_status "Updated $dir_name successfully."
        else
            print_warning "Could not update $dir_name. It may be modified locally."
        fi
        cd ..
    else
        print_status "Cloning $repo_name..."
        
        # Check if HF_TOKEN is set for private repos
        if [ -n "$HF_TOKEN" ]; then
            print_status "Using HuggingFace token for authentication..."
            git clone "https://USER:${HF_TOKEN}@huggingface.co/${repo_url}"
        else
            git clone "https://huggingface.co/${repo_url}"
        fi
        
        if [ $? -ne 0 ]; then
            print_error "Failed to clone $repo_name. Please check your internet connection and permissions."
            print_warning "If this is a private model, you need to set the HF_TOKEN environment variable."
            return 1
        fi
    fi
    
    # Verify the repository was cloned and contains essential files
    if [ -d "$dir_name" ]; then
        local file_count=$(find "$dir_name" -type f | wc -l)
        if [ "$file_count" -lt 5 ]; then
            print_warning "$dir_name contains very few files ($file_count). LFS content might not have been downloaded."
            print_warning "To fix, try: cd $dir_name && git lfs pull"
        else
            print_status "$dir_name cloned or updated successfully with $file_count files."
        fi
    else
        print_error "Failed to find or create $dir_name directory."
        return 1
    fi
    
    return 0
}

# Clone models with retry mechanism
clone_with_retry() {
    local repo_name=$1
    local repo_url=$2
    local dir_name=$3
    local attempts=3
    
    for ((i=1; i<=attempts; i++)); do
        if clone_model "$repo_name" "$repo_url" "$dir_name"; then
            return 0
        else
            if [ $i -lt $attempts ]; then
                print_warning "Attempt $i failed. Retrying in 3 seconds..."
                sleep 3
            else
                print_error "Failed to clone $repo_name after $attempts attempts."
                return 1
            fi
        fi
    done
}

# Check for HF_TOKEN in environment
if [ -z "$HF_TOKEN" ]; then
    # Try to read from .env file in the project root
    if [ -f "../../.env" ]; then
        print_status "Found .env file, checking for HF_TOKEN..."
        if grep -q "HF_TOKEN" "../../.env"; then
            export HF_TOKEN=$(grep "HF_TOKEN" "../../.env" | cut -d '=' -f2)
            print_status "HF_TOKEN loaded from .env file."
        fi
    else
        print_warning "No HF_TOKEN found. Some models may require authentication."
        print_warning "You can add it to a .env file in the project root or set it as an environment variable."
    fi
fi

# Clone models (removed deprecated qwen/janus models, kept QwenCaptioner)
clone_with_retry "Florence-2-base" "microsoft/Florence-2-base" "Florence-2-base"
clone_with_retry "BLIP model" "Salesforce/blip-image-captioning-base" "blip-image-captioning-base"
clone_with_retry "QwenCaptioner model" "Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed" "Qwen2.5-VL-7B-Captioner-Relaxed"

print_section "Summary"
print_status "All models cloned successfully!"
print_status "Model weights are in: $(pwd)"
print_status "You may need to update the model paths in the code to point to these local repositories."

# Return to original directory  
cd ../..