#!/bin/bash
# Download GGUF models for Ollama rerankers
# This script downloads quantized models from Hugging Face

set -e

echo "📥 Downloading Reranker GGUF Models..."
echo "======================================"

# Create models directory
mkdir -p models
cd models

# Function to download with progress
download_model() {
    local url=$1
    local filename=$2
    local description=$3
    
    echo ""
    echo "📦 Downloading $description..."
    echo "   File: $filename"
    
    if [ -f "$filename" ]; then
        echo "   ✅ Already exists, skipping"
        return 0
    fi
    
    if command -v wget &> /dev/null; then
        wget --progress=bar --show-progress -O "$filename" "$url"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$filename" "$url"
    else
        echo "   ❌ Error: Neither wget nor curl found. Please install one."
        return 1
    fi
    
    echo "   ✅ Download complete"
}

# BGE Models
echo ""
echo "🔵 BGE Reranker Models (Direct Scoring)"
download_model "https://huggingface.co/BAAI/bge-reranker-base/resolve/main/gguf/model.gguf" \
    "bge-reranker-base-q4_k_m.gguf" "BGE Base (~67MB)"

download_model "https://huggingface.co/BAAI/bge-reranker-large/resolve/main/gguf/model.gguf" \
    "bge-reranker-large-q4_k_m.gguf" "BGE Large (~334MB)"

download_model "https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/gguf/model.gguf" \
    "bge-reranker-v2-m3-Q4_K_M.gguf" "BGE V2-M3 (~559MB)"

# Qwen3 Models  
echo ""
echo "🟡 Qwen3 Reranker Models (Binary Classification)"
download_model "https://huggingface.co/Qwen/Qwen3-Reranker-0.6B-GGUF/resolve/main/Qwen3-Reranker-0.6B.Q4_K_M.gguf" \
    "Qwen3-Reranker-0.6B.Q4_K_M.gguf" "Qwen3 0.6B (~382MB)"

download_model "https://huggingface.co/Qwen/Qwen3-Reranker-4B-GGUF/resolve/main/Qwen3-Reranker-4B.Q4_K_M.gguf" \
    "Qwen3-Reranker-4B.Q4_K_M.gguf" "Qwen3 4B (~2.4GB)"

download_model "https://huggingface.co/Qwen/Qwen3-Reranker-8B-GGUF/resolve/main/Qwen3-Reranker-8B.Q4_K_M.gguf" \
    "Qwen3-Reranker-8B.Q4_K_M.gguf" "Qwen3 8B (~4.7GB)"

cd ..

echo ""
echo "🎉 Download Complete!"
echo "===================="
echo ""
echo "📊 Downloaded Models:"
ls -lh models/ 2>/dev/null || echo "No models directory found"

echo ""
echo "📁 Total size:"
du -sh models/ 2>/dev/null || echo "No models directory found"

echo ""
echo "▶️  Next steps:"
echo "   1. Run: ./setup_models.sh (create Ollama models)"
echo "   2. Run: ./validate_models.sh (test functionality)"
echo "   3. See MODEL_SETUP.md for usage examples"
