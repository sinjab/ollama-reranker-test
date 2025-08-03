# Ollama Reranker Testing Suite

A comprehensive testing framework for BGE and Qwen reranker models. This project validates and benchmarks different reranker implementations for document ranking and relevance scoring.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ollama-reranker-test
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Run the test suite**:
   ```bash
   # Test all models and implementations
   uv run python test_reranker.py
   
   # Test specific model type
   uv run python test_reranker.py --model-type bge
   uv run python test_reranker.py --model-type qwen
   
   # Test specific implementation
   uv run python test_reranker.py --implementation official
   uv run python test_reranker.py --implementation ollama
   ```

## 📊 Supported Models

### BGE Rerankers
| Model | Performance | Use Case |
|-------|-------------|----------|
| `BAAI/bge-reranker-v2-m3` | High | Production workloads |
| `BAAI/bge-reranker-base` | Balanced | General purpose |
| `BAAI/bge-reranker-large` | Maximum | High accuracy needs |

### Qwen Rerankers
| Model | Performance | Use Case |
|-------|-------------|----------|
| `Qwen/Qwen3-Reranker-0.6B` | High | Complex reasoning |
| `Qwen/Qwen3-Reranker-4B` | Higher | Advanced reasoning |
| `Qwen/Qwen3-Reranker-8B` | Maximum | Best performance |

## 🧪 Test Cases

The framework includes comprehensive test cases covering:

- **Basic Functionality**: Simple query-document ranking
- **Domain-Specific**: Machine learning, cooking, geography
- **Edge Cases**: Empty documents, invalid data, single documents
- **Performance**: Execution time and memory usage tracking

### Test Case Format

```json
{
  "query": "What is machine learning?",
  "documents": [
    "Machine learning is a subset of artificial intelligence.",
    "Cooking is the art of preparing food.",
    "Deep learning uses neural networks."
  ],
  "top_n": 2,
  "instruction": "Find AI topics"
}
```

## 📈 Sample Results

```
🤖 UNIFIED RERANKER TEST FRAMEWORK
==================================================

🔧 Testing BGE OFFICIAL: BAAI/bge-reranker-v2-m3
============================================================

📋 Testing: test_ml
Query: What is machine learning?
Documents: 3
✅ SUCCESS (0.074s)
📈 Rankings:
  1. Machine learning is a subset of artificial intelli... (score: 0.9994)
  2. Deep learning uses neural networks.... (score: 0.0017)

📊 TEST SUMMARY
==================================================
Total Tests: 18
Successful Tests: 18
Success Rate: 100.0%
✅ Tests completed successfully
```

## 📁 Project Structure

```
ollama-reranker-test/
├── test_reranker.py          # Unified test framework
├── compare_results.py        # Results comparison tool
├── tests/                   # Test case definitions
│   ├── test_basic.json
│   ├── test_capital.json
│   ├── test_cooking.json
│   ├── test_empty.json
│   ├── test_invalid.json
│   ├── test_ml.json
│   └── test_simple.json
├── results/                 # Generated test results
├── pyproject.toml          # Project configuration
└── LICENSE                 # MIT License
```

## ⚙️ Configuration

### Model Settings
- **Normalization**: `normalize=True` for 0-1 confidence scores
- **FP16**: `use_fp16=True` for performance optimization
- **Caching**: Models are cached to avoid repeated downloads

### Adding New Models

Update the `MODEL_CONFIGS` dictionary in `test_reranker.py`:

```python
MODEL_CONFIGS = {
    'new_model_type': {
        'official': {
            'models': ["path/to/model"],
            'default': "path/to/model"
        },
        'ollama': {
            'models': ["ollama_model_name"],
            'default': "ollama_model_name"
        }
    }
}
```

## 📊 Performance Benchmarks

| Model | Avg. Load Time | Avg. Score Time | Memory Usage |
|-------|----------------|-----------------|--------------|
| BGE v2-m3 | ~3.5s | ~0.07s | High |
| BGE base | ~1.0s | ~0.04s | Medium |
| BGE large | ~1.8s | ~0.04s | High |
| Qwen 0.6B | ~2.0s | ~0.05s | Medium |

## 🐛 Troubleshooting

### Common Issues

1. **Model Download Failures**:
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/huggingface/hub/
   uv run python test_reranker.py
   ```

2. **Memory Issues**:
   - Use smaller models (base instead of large)
   - Monitor system memory usage

3. **Dependency Conflicts**:
   ```bash
   # Recreate environment
   rm -rf .venv
   uv sync
   ```

### Error Messages

- `ModuleNotFoundError`: Run `uv sync` to install dependencies
- `CUDA out of memory`: Use CPU-only mode or smaller models
- API connection errors: Ensure Ollama server is running for API tests

## 🤝 Contributing

1. **Add Test Cases**: Create new JSON files in `tests/` directory
2. **Improve Error Handling**: Add robust error recovery
3. **Performance Optimization**: Optimize model loading and scoring
4. **Documentation**: Update README and add inline comments

## 📚 API Reference

### Output Format

```python
{
    "success": True,
    "results": [
        {
            "index": 0,
            "document": "Document text...",
            "relevance_score": 0.9994,
            "raw_response": "0.9994"
        }
    ],
    "time": 0.074,
    "error": None
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [BAAI](https://github.com/FlagOpen/FlagEmbedding) for the BGE reranker models
- [Qwen Team](https://github.com/QwenLM/Qwen) for the Qwen reranker models
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) for the Python library
- [uv](https://docs.astral.sh/uv/) for fast Python package management
