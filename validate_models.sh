#!/bin/bash
# Quick validation script for reranker models

echo "🔍 Validating Reranker Models..."
echo "================================="

# Test query and documents
QUERY="What is machine learning?"
DOCS='["Machine learning is a subset of AI", "Pizza recipe ingredients", "Deep learning uses neural networks"]'

# Function to test model
test_model() {
    local model=$1
    echo ""
    echo "🧪 Testing $model..."
    
    response=$(curl -s -X POST http://localhost:11434/api/rerank \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model\", \"query\": \"$QUERY\", \"documents\": $DOCS}")
    
    if echo "$response" | jq -e '.results | length > 0' > /dev/null 2>&1; then
        echo "✅ $model: Working correctly"
        echo "$response" | jq -r '.results[] | "   Doc \(.index): Score \(.relevance_score)"'
    else
        echo "❌ $model: Failed or no results"
        echo "   Response: $response"
    fi
}

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo "⚠️  jq not found. Install with: brew install jq"
    echo "   Results will be shown as raw JSON"
fi

# Test all models
test_model "bge-reranker-base"
test_model "bge-reranker-large" 
test_model "bge-reranker-v2-m3"
test_model "qwen3-reranker-0.6b"
test_model "qwen3-reranker-4b"
test_model "qwen3-reranker-8b"

echo ""
echo "🎯 Validation complete!"
