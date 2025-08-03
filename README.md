# Unified Reranker Testing Suite

A comprehensive testing framework for BGE and Qwen reranker models. This project validates and benchmarks different reranker implementations for document ranking and relevance scoring.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd bge-reranker-test
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Run the unified test suite**:
   ```bash
   # Test all models and implementations
   uv run python test_reranker.py
   
   # Test specific model type
   uv run python test_reranker.py --model-type bge
   uv run python test_reranker.py --model-type qwen
   
   # Test specific implementation
   uv run python test_reranker.py --implementation official
   uv run python test_reranker.py --implementation ollama
   
   # Test specific model
   uv run python test_reranker.py --model BAAI/bge-reranker-v2-m3
   uv run python test_reranker.py --model qwen_reranker_v2
   ```

## 📊 Models Tested

The test suite validates the following reranker models:

### BGE Rerankers
| Model | Type | Performance | Use Case |
|-------|------|-------------|----------|
| `BAAI/bge-reranker-v2-m3` | Normal | High | Production workloads |
| `BAAI/bge-reranker-base` | Normal | Balanced | General purpose |
| `BAAI/bge-reranker-large` | Normal | Maximum | High accuracy needs |

### Qwen Rerankers
| Model | Type | Performance | Use Case |
|-------|------|-------------|----------|
| `Qwen/Qwen3-Reranker-0.6B` | LLM-based | High | Complex reasoning |

> **⚠️ Note**: The `BAAI/bge-reranker-v2-gemma` model has been excluded due to a known bug in the FlagEmbedding library that causes `nan` scores.

## 🏗️ Architecture

### Unified Framework Benefits

**Before (4 separate files)**:
- `test_official_bge.py` - 220 lines
- `test_official_qwen.py` - 259 lines  
- `test_ollama_bge.py` - 185 lines
- `test_ollama_qwen.py` - 185 lines
- **Total**: 849 lines with massive duplication

**After (1 unified file)**:
- `test_reranker.py` - 400 lines
- **Savings**: 53% reduction in code, 100% reduction in duplication

### Key Improvements

1. **Single Source of Truth**: All test logic in one place
2. **Configurable Testing**: Command-line arguments for flexible testing
3. **Consistent Output**: Standardized result formats
4. **Easy Maintenance**: Changes only need to be made once
5. **Backward Compatibility**: Wrapper scripts maintain old interface

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
  3. The weather today is sunny.... (score: 0.0000)

📊 TEST SUMMARY
==================================================
📊 bge_official_BAAI_bge-reranker-v2-m3:
  Total Tests: 6
  Successful Tests: 6
  Success Rate: 100.0%

📊 OVERALL SUMMARY
========================================
Total Tests: 18
Successful Tests: 18
Success Rate: 100.0%
✅ Tests completed successfully
```

## 🛠️ Development

### Environment Setup

```bash
# Install all dependencies
uv sync

# Install development dependencies
uv sync --extra dev

# Activate virtual environment
uv shell
```

### Adding New Models

To add a new model type, simply update the `MODEL_CONFIGS` dictionary in `test_reranker.py`:

```python
MODEL_CONFIGS = {
    'new_model_type': {
        'official': {
            'models': ["path/to/model"],
            'default': "path/to/model"
        },
        'ollama': {
            'default': "ollama_model_name"
        }
    }
}
```

## 📁 Project Structure

```
bge-reranker-test/
├── test_reranker.py          # 🆕 Unified test framework
├── compare_results.py        # 📊 Results comparison tool
├── tests/                   # 📋 Test case definitions
│   ├── test_basic.json
│   ├── test_capital.json
│   ├── test_cooking.json
│   ├── test_empty.json
│   ├── test_invalid.json
│   ├── test_ml.json
│   └── test_simple.json
├── results/                 # 📈 Generated test results
│   ├── bge_official_*.json
│   ├── qwen_official_*.json
│   ├── bge_ollama_*.json
│   └── qwen_ollama_*.json
└── pyproject.toml          # 📦 Project configuration
```

## �� Configuration

### Model Settings

All models use the following configuration:
- **Normalization**: `normalize=True` for 0-1 confidence scores
- **FP16**: `use_fp16=True` for performance optimization
- **Caching**: Models are cached to avoid repeated downloads

### Test Parameters

- **Batch Processing**: Documents are processed in pairs
- **Error Handling**: Graceful fallback for failed models
- **Progress Tracking**: Real-time execution time and status updates
- **Result Storage**: JSON output for analysis and comparison

## 📊 Performance Benchmarks

| Model | Avg. Load Time | Avg. Score Time | Memory Usage |
|-------|----------------|-----------------|--------------|
| v2-m3 | ~3.5s | ~0.07s | High |
| base | ~1.0s | ~0.04s | Medium |
| large | ~1.8s | ~0.04s | High |

## 🐛 Troubleshooting

### Common Issues

1. **Model Download Failures**:
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/huggingface/hub/models--BAAI--*
   uv run python test_reranker.py
   ```

2. **Memory Issues**:
   - Use smaller models (base instead of large)
   - Reduce batch sizes in test cases
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
- `nan scores`: Excluded gemma model - use normal BGE rerankers

## 🤝 Contributing

1. **Add Test Cases**: Create new JSON files in `tests/` directory
2. **Improve Error Handling**: Add robust error recovery
3. **Performance Optimization**: Optimize model loading and scoring
4. **Documentation**: Update README and add inline comments

### Test Case Guidelines

- Include diverse query types (technical, general, edge cases)
- Provide realistic document sets
- Test both single and multiple document scenarios
- Include edge cases (empty arrays, special characters)

## 📚 API Reference

### Main Functions

```python
def load_real_model(model_name: str) -> Tuple[Dict, Optional[str]]:
    """Load a BGE reranker model."""
    
def test_official_bge(test_case: Dict, model_info: Dict) -> Dict:
    """Run a single test case with a specific model."""
    
def load_test_cases() -> List[Dict]:
    """Load all test cases from JSON files."""
```

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
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) for the Python library
- [uv](https://docs.astral.sh/uv/) for fast Python package management

---

**Built with ❤️ for reliable BGE reranker testing**
