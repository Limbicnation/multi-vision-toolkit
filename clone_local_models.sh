#!/bin/bash
# Script to clone models directly from Hugging Face into the local models directory

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
            print_error "Could not automatically install git-lfs. Please install it manually."
            exit 1
        fi
    elif [[ "$PLATFORM" == "macos" ]]; then
        if command -v brew &> /dev/null; then
            print_status "Using Homebrew to install git-lfs..."
            brew install git-lfs
        else
            print_error "Could not automatically install git-lfs. Please install Homebrew first."
            exit 1
        fi
    elif [[ "$PLATFORM" == "windows" ]]; then
        print_error "Please install git-lfs manually on Windows from https://git-lfs.github.com/"
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
mkdir -p models/weights

# Set variables
MODEL_DIR="models/weights"
QWEN_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_LOCAL_DIR="$MODEL_DIR/Qwen2.5-VL-3B-Instruct"

# Clone Qwen model specifically for local usage
clone_qwen_model() {
    print_section "Cloning Qwen2.5-VL-3B-Instruct model for local storage"
    
    if [ -d "$QWEN_LOCAL_DIR" ]; then
        print_status "Qwen model directory already exists at $QWEN_LOCAL_DIR"
        print_status "Checking for complete files..."
        
        # Check for essential files
        if [ -f "$QWEN_LOCAL_DIR/config.json" ] && \
           [ -f "$QWEN_LOCAL_DIR/tokenizer.json" ] && \
           [ -f "$QWEN_LOCAL_DIR/model.safetensors.index.json" ]; then
            print_status "Essential model files found."
            
            # Check for model weight files
            if ls "$QWEN_LOCAL_DIR"/model-*.safetensors &> /dev/null; then
                print_status "Model weight files found. Qwen model appears to be complete."
                return 0
            else
                print_warning "No model weight files found. Model may be incomplete."
                print_warning "Attempting to pull LFS files..."
                
                # Try to pull LFS files
                (cd "$QWEN_LOCAL_DIR" && git lfs pull)
                
                # Check again after pull
                if ls "$QWEN_LOCAL_DIR"/model-*.safetensors &> /dev/null; then
                    print_status "Successfully pulled model weight files."
                    return 0
                else
                    print_warning "Still missing model files. Removing incomplete directory and re-downloading..."
                    rm -rf "$QWEN_LOCAL_DIR"
                fi
            fi
        else
            print_warning "Essential model files missing. Removing incomplete directory and re-downloading..."
            rm -rf "$QWEN_LOCAL_DIR"
        fi
    fi
    
    print_status "Cloning Qwen model to $QWEN_LOCAL_DIR..."
    
    # Check for HF_TOKEN in environment
    if [ -n "$HF_TOKEN" ]; then
        print_status "Using HuggingFace token for authentication..."
        git clone --filter=blob:none "https://USER:${HF_TOKEN}@huggingface.co/${QWEN_MODEL}" "$QWEN_LOCAL_DIR"
    else
        git clone --filter=blob:none "https://huggingface.co/${QWEN_MODEL}" "$QWEN_LOCAL_DIR"
    fi
    
    # Check if clone was successful
    if [ $? -ne 0 ]; then
        print_error "Failed to clone model using git. Make sure you have git-lfs installed."
        print_error "Alternatively, try setting the HF_TOKEN environment variable if the model is private."
        return 1
    fi
    
    # Pull LFS objects
    print_status "Pulling LFS objects for model weights..."
    (cd "$QWEN_LOCAL_DIR" && git lfs pull)
    
    # Verify model files were downloaded
    print_status "Verifying model files..."
    if [ -f "$QWEN_LOCAL_DIR/config.json" ] && \
       [ -f "$QWEN_LOCAL_DIR/tokenizer.json" ] && \
       ls "$QWEN_LOCAL_DIR"/model-*.safetensors &> /dev/null; then
        print_status "Model files verified. Ready to use!"
        
        # Count files and show file sizes
        file_count=$(find "$QWEN_LOCAL_DIR" -type f | wc -l)
        total_size=$(du -sh "$QWEN_LOCAL_DIR" | cut -f1)
        print_status "Downloaded $file_count files, total size: $total_size"
    else
        print_warning "Model might be incomplete. Checking individual files..."
        ls -la "$QWEN_LOCAL_DIR"
        
        # Check for specific model files
        if [ ! -f "$QWEN_LOCAL_DIR/config.json" ]; then
            print_error "Missing config.json"
        fi
        if [ ! -f "$QWEN_LOCAL_DIR/tokenizer.json" ]; then
            print_error "Missing tokenizer.json"
        fi
        if [ ! -f "$QWEN_LOCAL_DIR/model.safetensors.index.json" ]; then
            print_error "Missing model.safetensors.index.json"
        fi
        
        # Check for model shards
        model_shards=$(find "$QWEN_LOCAL_DIR" -name "model-*.safetensors" | wc -l)
        if [ "$model_shards" -eq 0 ]; then
            print_error "No model weight files found. Download failed."
        else
            print_warning "Found only $model_shards model weight files. Model may be incomplete."
        fi
        
        return 1
    fi
}

print_section "Multi-Vision Toolkit Local Model Download"
print_status "This script will clone the Qwen model directly into your local models/weights directory."
print_status "This avoids relying on Hugging Face's cache system, which might be cleared."

# Check for HF_TOKEN in environment or .env file
if [ -z "$HF_TOKEN" ]; then
    # Try to read from .env file in the project root
    if [ -f ".env" ]; then
        print_status "Found .env file, checking for HF_TOKEN..."
        if grep -q "HF_TOKEN" ".env"; then
            export HF_TOKEN=$(grep "HF_TOKEN" ".env" | cut -d '=' -f2)
            print_status "HF_TOKEN loaded from .env file."
        fi
    else
        print_warning "No HF_TOKEN found. Some models may require authentication."
        print_warning "You can add it to a .env file in the project root or set it as an environment variable."
    fi
fi

# Clone Qwen model
clone_qwen_model

# Create an environment setup script
cat > ".env.local" << EOF
# Environment variables for using local model storage
# Source this file to use local models: source .env.local

export TRANSFORMERS_CACHE="$(pwd)/$MODEL_DIR"
export HF_HOME="$(pwd)/$MODEL_DIR"
export HF_HUB_CACHE="$(pwd)/$MODEL_DIR"
export TORCH_HOME="$(pwd)/$MODEL_DIR"

# Uncomment to force offline mode (use only local files)
# export HF_HUB_OFFLINE=1

echo "Environment set up for local model usage."
echo "Model directory: $(pwd)/$MODEL_DIR"
EOF

print_section "Download Complete"
print_status "Model download script completed. The models are now in your local directory."
print_status "To use these models, run the toolkit with:"
print_status "python main.py --model qwen_local"
print_status ""
print_status "You can set up your environment to use local models with:"
print_status "source .env.local"