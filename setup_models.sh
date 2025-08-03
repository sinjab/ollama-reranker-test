#!/bin/bash
# Ollama Reranker Models Setup Script
# This script creates all reranker models in Ollama using the provided templates

set -e

echo "🚀 Setting up Ollama Reranker Models..."
echo "========================================="

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Error: Ollama is not running. Please start Ollama first:"
    echo "   ollama serve"
    exit 1
fi

echo "✅ Ollama is running"

# Function to create model
create_model() {
    local model_name=$1
    local template_file=$2
    
    echo "📦 Creating model: $model_name"
    if ollama create "$model_name" -f "$template_file"; then
        echo "✅ Successfully created $model_name"
    else
        echo "❌ Failed to create $model_name"
        return 1
    fi
}

# Create BGE models
echo ""
echo "🔵 Creating BGE Models (Direct Scoring)..."
create_model "bge-reranker-base" "templates/Modelfile.bge-base"
create_model "bge-reranker-large" "templates/Modelfile.bge-large"
create_model "bge-reranker-v2-m3" "templates/Modelfile.bge-v2-m3"

# Create Qwen3 models
echo ""
echo "🟡 Creating Qwen3 Models (Binary Classification)..."
create_model "qwen3-reranker-0.6b" "templates/Modelfile.qwen3-0.6b"
create_model "qwen3-reranker-4b" "templates/Modelfile.qwen3-4b"
create_model "qwen3-reranker-8b" "templates/Modelfile.qwen3-8b"

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📋 Created Models:"
ollama list | grep reranker || echo "No reranker models found (check for errors above)"

echo ""
echo "🧪 Test a model:"
echo 'curl -X POST http://localhost:11434/api/rerank \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"model": "bge-reranker-base", "query": "test", "documents": ["relevant doc", "irrelevant doc"]}'"'"

echo ""
echo "📖 For usage examples, see MODEL_SETUP.md"
