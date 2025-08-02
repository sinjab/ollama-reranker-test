# BGE Reranking Testing Suite

Comprehensive testing framework for BGE reranking implementation in Ollama.

## Quick Start

### Using uv (Recommended)

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Setup the environment**:
   ```bash
   uv sync
   ```

3. **Start Ollama with new engine**:
   ```bash
   OLLAMA_NEW_ENGINE=1 ./ollama serve
   ```

4. **Setup test environment**:
   ```bash
   bash scripts/setup_test_env.sh
   ```

5. **Run comprehensive tests**:
   ```bash
   uv run python scripts/run_comprehensive_tests.py
   ```

### Using pip (Legacy)

1. `pip install -r requirements.txt`
2. `OLLAMA_NEW_ENGINE=1 ./ollama serve`
3. `bash scripts/setup_test_env.sh`
4. `python scripts/run_comprehensive_tests.py`

## Development Setup

### With uv (Recommended)

```bash
# Install development dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=utils --cov=tests

# Format code
uv run black .
uv run isort .

# Type checking
uv run mypy .

# Linting
uv run ruff check .
```

### Common uv Commands

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Run a command in the virtual environment
uv run python script.py

# Activate the virtual environment
uv shell

# Update dependencies
uv lock --upgrade
```
