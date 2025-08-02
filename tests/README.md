# Test Cases

This directory contains JSON test files for the Qwen3-Reranker comparison tests.

## Test File Format

Each test file should be named `test_*.json` and contain the following structure:

```json
{
  "query": "What is the capital of China?",
  "documents": [
    "The capital of China is Beijing.",
    "China is a large country in Asia.",
    "Paris is the capital of France."
  ],
  "instruction": "Find AI topics",  // Optional
  "top_n": 2,                      // Optional
  "model": "qwen_reranker_v2"      // Optional
}
```

## Required Fields

- `query`: The search query string
- `documents`: Array of document strings to rank

## Optional Fields

- `instruction`: Additional instruction for the reranker
- `top_n`: Limit number of results returned
- `model`: Specific model to use (defaults to "qwen_reranker_v2")

## Test Cases

### test_capital.json
Tests basic reranking with a simple query about capital cities.

### test_ml.json
Tests reranking with an instruction to find AI-related content.

### test_cooking.json
Tests reranking with a top_n limit to return only the best matches.

### test_empty.json
Tests error handling with empty documents array.

### test_invalid.json
Tests error handling with an invalid model name.

## Adding New Tests

To add a new test case:

1. Create a new JSON file named `test_<name>.json`
2. Follow the format above
3. The test will be automatically loaded by `compare_test.py`

## Manual Testing

You can test individual files manually using curl:

```bash
curl -X POST http://localhost:11434/api/rerank \
  -H "Content-Type: application/json" \
  -d @tests/test_capital.json
``` 