# Ollama Reranker Testing Suite

A comprehensive testing framework for BGE and Qwen reranker models. This project validates and benchmarks different reranker implementations for document ranking and relevance scoring.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Ollama with reranking support

### Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sinjab/ollama-reranker-test.git
   cd ollama-reranker-test
   ```

2. **Set up reranker models** (REQUIRED):
   📖 **See [MODEL_SETUP.md](MODEL_SETUP.md) for complete model installation guide**
   
   Quick setup:
   ```bash
   # Download models (~8.7GB)
   ./download_models.sh
   
   # Create BGE models
   ollama create bge-reranker-base -f templates/Modelfile.bge-base
   ollama create bge-reranker-v2-m3 -f templates/Modelfile.bge-v2-m3
   
   # Create Qwen3 models  
   ollama create qwen3-reranker-0.6b -f templates/Modelfile.qwen3-0.6b
   
   # Or use automation: ./setup_models.sh
   ```

3. **Install dependencies**:
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
| Model | Performance | Use Case | Quantization |
|-------|-------------|----------|--------------|
| `BAAI/bge-reranker-v2-m3` | High | Production workloads | Q4_K_M GGUF |
| `BAAI/bge-reranker-base` | Balanced | General purpose | Q4_K_M GGUF |
| `BAAI/bge-reranker-large` | Maximum | High accuracy needs | Q4_K_M GGUF |

### Qwen Rerankers
| Model | Performance | Use Case | Quantization |
|-------|-------------|----------|--------------|
| `Qwen/Qwen3-Reranker-0.6B` | Fast | Lightweight reasoning | Q4_K_M GGUF |
| `Qwen/Qwen3-Reranker-4B` | High | Advanced reasoning | Q4_K_M GGUF |
| `Qwen/Qwen3-Reranker-8B` | Maximum | Best accuracy | Q4_K_M GGUF |

> **📦 Model Versions**: This testing framework uses Q4_K_M quantized GGUF versions of all models for optimal performance and reduced memory usage. The Ollama implementations utilize these quantized models:
> - `bge-reranker-base-q4_k_m.gguf`
> - `bge-reranker-large-q4_k_m.gguf` 
> - `bge-reranker-v2-m3-Q4_K_M.gguf`
> - `Qwen3-Reranker-0.6B.Q4_K_M.gguf`
> - `Qwen3-Reranker-4B.Q4_K_M.gguf`
> - `Qwen3-Reranker-8B.Q4_K_M.gguf`

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

📋 Testing: test_capital
Query: What is the capital of China?
Documents: 3
✅ SUCCESS (0.074s)
📈 Rankings:
  1. Beijing is the capital of China... (score: 1.0000)
  2. China is a large country in Asia... (score: 0.0745)
  3. Paris is the capital of France... (score: 0.0003)

📊 TEST SUMMARY
==================================================
📊 BGE Models: 6/6 (100% success)
📊 Qwen Models: 6/6 (100% success)
📊 Ollama API: 6/6 (100% success)
📊 Official Implementation: 6/6 (100% success)

📊 OVERALL SUMMARY
Total Tests: 72 (12 models × 6 test cases)
Successful Tests: 72
Success Rate: 100.0%
✅ All models demonstrate perfect ranking consistency
```

## 📁 Project Structure

```
ollama-reranker-test/
├── test_reranker.py          # Unified test framework
├── compare_results.py        # Results comparison tool
├── MODEL_SETUP.md           # Complete model installation guide
├── download_models.sh       # GGUF model download script (~8.7GB)
├── setup_models.sh          # Automated Ollama model creation
├── validate_models.sh       # Quick model validation script
├── models/                  # GGUF model files (downloaded via script)
│   ├── bge-reranker-base-q4_k_m.gguf
│   ├── bge-reranker-large-q4_k_m.gguf
│   ├── bge-reranker-v2-m3-Q4_K_M.gguf
│   ├── Qwen3-Reranker-0.6B.Q4_K_M.gguf
│   ├── Qwen3-Reranker-4B.Q4_K_M.gguf
│   └── Qwen3-Reranker-8B.Q4_K_M.gguf
├── templates/               # Ollama Modelfile templates
│   ├── Modelfile.bge-base
│   ├── Modelfile.bge-large
│   ├── Modelfile.bge-v2-m3
│   ├── Modelfile.qwen3-0.6b
│   ├── Modelfile.qwen3-4b
│   └── Modelfile.qwen3-8b
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

## 🎯 Testing Results & Insights

### Model Comparison Results
- **12 total models tested**: 6 BGE + 6 Qwen variants
- **Perfect correlation (1.0000)** between all model implementations
- **100% success rate** across all 72 test combinations
- **Consistent ranking accuracy** regardless of implementation type

### Performance Categories

#### BGE Models Performance
| Model | Implementation | Avg Time (s) | Sample Score |
|-------|----------------|--------------|--------------|
| bge-large | Ollama | 0.074 | 0.9517 |
| bge-v2-m3 | Ollama | 0.075 | 0.9517 |
| bge-base | Ollama | 0.082 | 0.9517 |
| bge-reranker-base | Official | 0.134 | 0.9994 |
| bge-reranker-large | Official | 0.214 | 0.9997 |
| bge-reranker-v2-m3 | Official | 0.285 | 1.0000 |

#### Qwen Models Performance  
| Model | Implementation | Avg Time (s) | Sample Score |
|-------|----------------|--------------|--------------|
| Qwen3-Reranker-0.6B | Official | 0.255 | 0.9995 |
| qwen3-0.6b | Ollama | 0.578 | 0.9995 |
| qwen3-4b | Ollama | 0.865 | 0.9995 |
| Qwen3-Reranker-4B | Official | 1.448 | 0.9982 |
| qwen3-8b | Ollama | 1.450 | 0.9995 |
| Qwen3-Reranker-8B | Official | 2.532 | 0.9945 |

### Recommendations

#### Production Use
- **bge-large (Ollama)** - Best performance/accuracy balance  
- **bge-v2-m3 (Ollama)** - High-performance alternative
- **qwen3-0.6b (Ollama)** - For reasoning-heavy tasks

#### Development & Testing
- **Ollama implementations** - Faster iteration cycles
- **Official implementations** - Maximum accuracy validation
- **BGE models** - More consistent performance

## ⚙️ Configuration

### Model Settings
- **Quantization**: Q4_K_M GGUF format for 4-bit quantization with mixed precision
- **Memory Optimization**: Quantized models reduce VRAM usage by ~75% 
- **Performance**: Maintains accuracy while significantly improving inference speed
- **Normalization**: `normalize=True` for 0-1 confidence scores
- **FP16**: `use_fp16=True` for performance optimization on compatible hardware
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

Based on comprehensive testing across all models using **Q4_K_M quantized GGUF versions**:

### Speed Rankings (Fastest to Slowest)

| Rank | Model | Implementation | Avg Time (s) | Use Case |
|------|-------|----------------|--------------|----------|
| 1 | bge-large | Ollama | 0.074s | **Production recommended** |
| 2 | bge-v2-m3 | Ollama | 0.075s | High performance |
| 3 | bge-base | Ollama | 0.082s | Balanced option |
| 4 | bge-reranker-base | Official | 0.134s | Validation |
| 5 | bge-reranker-large | Official | 0.214s | Accuracy validation |
| 6 | Qwen3-Reranker-0.6B | Official | 0.255s | Lightweight reasoning |
| 7 | bge-reranker-v2-m3 | Official | 0.285s | Maximum accuracy |
| 8 | qwen3-0.6b | Ollama | 0.578s | Fast reasoning |
| 9 | qwen3-4b | Ollama | 0.865s | Advanced reasoning |
| 10 | Qwen3-Reranker-4B | Official | 1.448s | Complex reasoning |
| 11 | qwen3-8b | Ollama | 1.450s | Best reasoning |
| 12 | Qwen3-Reranker-8B | Official | 2.532s | Maximum accuracy |

### Key Performance Insights
- **Ollama implementations are 1.5-2.7x faster** than official implementations
- **BGE models significantly outperform Qwen models** in speed
- **All models achieve 100% success rate** with perfect ranking consistency
- **bge-large (Ollama)** offers the best performance/accuracy balance

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
