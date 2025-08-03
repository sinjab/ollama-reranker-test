# 🤖 Reranker Model Setup Guide

This directory contains quantized GGUF models and Ollama templates for BGE and Qwen3 reranker models, providing a complete local reranking solution.

## 📦 Available Models

### BGE Rerankers (Direct Scoring)
| Model | File | Size | Use Case |
|-------|------|------|----------|
| BGE Base | `bge-reranker-base-q4_k_m.gguf` | ~67MB | General purpose, fast |
| BGE Large | `bge-reranker-large-q4_k_m.gguf` | ~334MB | Higher accuracy |
| BGE V2-M3 | `bge-reranker-v2-m3-Q4_K_M.gguf` | ~559MB | Best performance |

### Qwen3 Rerankers (Binary Classification)
| Model | File | Size | Use Case |
|-------|------|------|----------|
| Qwen3 0.6B | `Qwen3-Reranker-0.6B.Q4_K_M.gguf` | ~382MB | Lightweight reasoning |
| Qwen3 4B | `Qwen3-Reranker-4B.Q4_K_M.gguf` | ~2.4GB | Advanced reasoning |
| Qwen3 8B | `Qwen3-Reranker-8B.Q4_K_M.gguf` | ~4.7GB | Maximum accuracy |

## 🚀 Quick Setup

### Prerequisites
- Ollama installed and running with reranking support
- At least 8GB RAM (16GB recommended for larger models)
- ~8GB free disk space for all models

### 1. Clone Repository
```bash
git clone https://github.com/sinjab/ollama-reranker-test.git
cd ollama-reranker-test
```

### 2. Create Models with Ollama
```bash
# BGE Models (Fast Direct Scoring)
ollama create bge-reranker-base -f templates/Modelfile.bge-base
ollama create bge-reranker-large -f templates/Modelfile.bge-large  
ollama create bge-reranker-v2-m3 -f templates/Modelfile.bge-v2-m3

# Qwen3 Models (Advanced Reasoning)
ollama create qwen3-reranker-0.6b -f templates/Modelfile.qwen3-0.6b
ollama create qwen3-reranker-4b -f templates/Modelfile.qwen3-4b
ollama create qwen3-reranker-8b -f templates/Modelfile.qwen3-8b
```

### 3. Verify Installation
```bash
# List created models
ollama list | grep reranker

# Test a model
curl -X POST http://localhost:11434/api/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-base",
    "query": "machine learning",
    "documents": ["AI and ML are related", "Pizza recipe", "Neural networks"]
  }'
```

## 🎯 Usage Examples

### BGE Reranker Usage
```bash
# Simple query-document ranking
curl -X POST http://localhost:11434/api/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-v2-m3",
    "query": "What is artificial intelligence?",
    "documents": [
      "AI is machine intelligence that mimics human cognitive functions",
      "Cooking pasta requires boiling water first",
      "Machine learning is a subset of artificial intelligence", 
      "The weather today is sunny and warm"
    ]
  }'

# Expected Response:
{
  "model": "bge-reranker-v2-m3",
  "results": [
    {"index": 0, "relevance_score": 0.9517},
    {"index": 2, "relevance_score": 0.0517},
    {"index": 3, "relevance_score": 0.0001},
    {"index": 1, "relevance_score": 0.0001}
  ]
}
```

### Qwen3 Reranker Usage
```bash
# Advanced reasoning with custom instruction
curl -X POST http://localhost:11434/api/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-reranker-0.6b",
    "query": "sustainable energy solutions",
    "documents": [
      "Solar panels convert sunlight into electricity efficiently",
      "Coal mining has significant environmental impacts",
      "Wind turbines generate clean renewable energy",
      "Nuclear fusion research shows promising results",
      "Fast food restaurants are expanding globally"
    ],
    "instruction": "Focus on clean and renewable energy technologies"
  }'

# Expected Response:
{
  "model": "qwen3-reranker-0.6b",
  "results": [
    {"index": 0, "relevance_score": 0.9995},
    {"index": 2, "relevance_score": 0.9995},
    {"index": 3, "relevance_score": 0.1631},
    {"index": 1, "relevance_score": 0.0035},
    {"index": 4, "relevance_score": 0.0001}
  ]
}
```

## 🔧 Model Characteristics

### BGE Models
- **Template Format**: Simple `{{ .Query }}{{ .Document }}`
- **Scoring Method**: Direct relevance scoring (0.0 to 1.0)
- **Performance**: Very fast, optimized for speed
- **Best For**: High-throughput applications, real-time ranking

### Qwen3 Models  
- **Template Format**: Chat-based with system/user/assistant structure
- **Scoring Method**: Binary classification simulation ("yes"/"no" → probability)
- **Performance**: Slower but more accurate reasoning
- **Best For**: Complex queries requiring nuanced understanding

## 📊 Performance Benchmarks

Based on comprehensive testing:

| Model | Avg Response Time | Accuracy | Memory Usage | Best Use Case |
|-------|------------------|----------|--------------|---------------|
| BGE Base | 74ms | High | ~500MB | Production speed |
| BGE Large | 75ms | Higher | ~800MB | Balanced performance |
| BGE V2-M3 | 82ms | Highest | ~1.2GB | Maximum BGE accuracy |
| Qwen3 0.6B | 578ms | Very High | ~1GB | Lightweight reasoning |
| Qwen3 4B | 865ms | Excellent | ~3GB | Advanced reasoning |
| Qwen3 8B | 1450ms | Best | ~5.5GB | Maximum accuracy |

## 🛠️ Advanced Configuration

### Custom Templates
You can modify the templates in the `templates/` directory to customize behavior:

```dockerfile
# Example: Custom BGE with instruction support
FROM ./models/bge-reranker-v2-m3-Q4_K_M.gguf
TEMPLATE """Query: {{ .Query }}
Context: {{ .Instruction }}
Document: {{ .Document }}"""
PARAMETER stop "</s>"
PARAMETER temperature 0.0
```

### Environment Variables
```bash
# For better performance
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2

# For debugging
export OLLAMA_DEBUG=1
```

## 🔍 Troubleshooting

### Common Issues

**Model Creation Fails:**
```bash
# Check if Ollama is running
ollama list

# Verify GGUF file exists
ls -la models/

# Try recreating the model
ollama rm bge-reranker-base
ollama create bge-reranker-base -f templates/Modelfile.bge-base
```

**API Errors:**
```bash
# Check Ollama server status
curl http://localhost:11434/api/tags

# Verify model is loaded
ollama show bge-reranker-base

# Test with simple request
curl -X POST http://localhost:11434/api/rerank \
  -d '{"model": "bge-reranker-base", "query": "test", "documents": ["test doc"]}'
```

**Memory Issues:**
- Use smaller models (BGE Base, Qwen3 0.6B) for limited memory
- Reduce `OLLAMA_NUM_PARALLEL` if experiencing OOM errors
- Monitor memory usage with `top` or `htop`

## 📚 Additional Resources

- **Test Scripts**: Run `python test_reranker.py` for comprehensive validation
- **Performance Comparison**: Use `python compare_results.py` for benchmarks
- **Official Documentation**: Check Ollama docs for latest API updates
- **Community**: Join discussions about reranking implementations

## 🤝 Contributing

Found issues or improvements? Please:
1. Test your changes with the provided test suite
2. Update documentation as needed
3. Submit pull requests with clear descriptions

---

**Note**: All models are quantized to Q4_K_M for optimal balance between performance and accuracy. Original model weights are from BAAI (BGE) and Qwen teams.
