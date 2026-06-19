# LLM Data Generator

A unified toolkit for collecting LLM responses and tracking token usage, latency, model metadata, and output records.

It supports online providers such as OpenAI, Anthropic, Google, HuggingFace-hosted models, and local models through either Transformers or OpenAI-compatible local servers.

---

## Features

- Unified interface for multiple LLM providers
- Token, latency, and response tracing
- YAML-based model configuration
- Batch processing and model comparison
- JSONL persistence for downstream analysis
- Optional cost estimation for known API models
- Local HuggingFace model support

---

## Quick Start

### 1. Install dependencies

```bash
cd data_generator
pip install -r requirements.txt
```

### 2. Choose your setup

Use online APIs:

```bash
cp .env.example .env
# Edit .env and fill in your API keys
python examples/01_quick_start.py
```

Use local HuggingFace models:

```bash
pip install transformers torch accelerate
python examples/02_local_model.py
```

---

## Basic Usage

```python
from generator import LLMDataGenerator

generator = LLMDataGenerator(config_path="config.yaml")

response = generator.generate_single(
    query="What is machine learning?",
    model_name="gpt-3.5-turbo",
    save_result=True,
)

print(f"Response: {response.response}")
print(f"Token Usage: {response.token_usage.total_tokens}")
print(f"Latency: {response.timing_info.duration_seconds:.2f}s")
```

### Local Model Usage

```python
from local_generator import LocalLLMGenerator

generator = LocalLLMGenerator(
    model_path="meta-llama/Meta-Llama-3.2-1B-Instruct",
    device="auto",
    verbose=True,
)

response = generator.generate_single(
    query="What is machine learning?",
    temperature=0.7,
    max_tokens=100,
    save_result=True,
)
```

### Batch Processing

```python
queries = [
    "What is deep learning?",
    "What is a neural network?",
    "What is reinforcement learning?",
]

responses = generator.generate_batch(
    queries=queries,
    model_name="gpt-3.5-turbo",
    save_results=True,
    show_progress=True,
)
```

### Model Comparison

```python
results = generator.compare_models(
    query="Explain quantum computing",
    model_names=["gpt-3.5-turbo", "claude-3-haiku", "gemini-pro"],
    save_results=True,
)
```

---

## Examples

| Example | Description | Best For |
| --- | --- | --- |
| [01_quick_start.py](examples/01_quick_start.py) | Minimal online API quick start | First-time users |
| [02_local_model.py](examples/02_local_model.py) | Run a local HuggingFace model | Offline use and cost saving |
| [03_batch_processing.py](examples/03_batch_processing.py) | Batch data generation | Large-scale collection |
| [04_model_comparison.py](examples/04_model_comparison.py) | Compare multiple models | Model selection |
| [05_data_analysis.py](examples/05_data_analysis.py) | Analyze and export collected data | Post-hoc analysis |

For more details, see [examples/README.md](examples/README.md).

---

## Supported Models

### Closed-Source APIs

| Provider | Example Models | Config Names |
| --- | --- | --- |
| OpenAI | GPT-4, GPT-3.5-turbo | `gpt-4`, `gpt-3.5-turbo` |
| Anthropic | Claude 3 series | `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku` |
| Google | Gemini series | `gemini-pro`, `gemini-2.5-pro`, `gemini-2.5-flash` |
| Azure | Azure OpenAI | `azure-gpt-4` |

### Open Models

| Model | Config Name | Notes |
| --- | --- | --- |
| Llama 3.1 8B | `llama3.1-8b` | 128K context |
| Llama 3.2 1B | `llama3.2-1b` | Small and fast for testing |
| Qwen 2.5 7B | custom | Strong multilingual performance |

For local deployment, see [docs/LOCAL_TRANSFORMERS_GUIDE.md](docs/LOCAL_TRANSFORMERS_GUIDE.md) and [docs/LOCAL_MODEL_GUIDE.md](docs/LOCAL_MODEL_GUIDE.md).

---

## Project Structure

```text
data_generator/
|-- README.md
|-- requirements.txt
|-- config.yaml
|-- .env.example
|-- generator.py
|-- local_generator.py
|-- models.py
|-- utils.py
|-- examples/
|   |-- README.md
|   |-- 01_quick_start.py
|   |-- 02_local_model.py
|   |-- 03_batch_processing.py
|   |-- 04_model_comparison.py
|   `-- 05_data_analysis.py
|-- docs/
|   |-- QUICKSTART.md
|   |-- LOCAL_TRANSFORMERS_GUIDE.md
|   `-- LOCAL_MODEL_GUIDE.md
`-- data/
    `-- README.md
```

---

## Data Format

Generated records are saved as JSONL, one JSON object per line:

```json
{
  "query_id": "20241215_123456_abc123",
  "query": "What is machine learning?",
  "response": "Machine learning is...",
  "model_info": {
    "model_name": "gpt-3.5-turbo",
    "provider": "openai",
    "temperature": 1.0
  },
  "token_usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  },
  "timing_info": {
    "start_time": "2024-12-15T12:34:56",
    "end_time": "2024-12-15T12:34:58",
    "duration_seconds": 2.5,
    "duration_ms": 2500.0
  },
  "timestamp": "2024-12-15T12:34:58",
  "success": true,
  "metadata": {
    "estimated_cost_usd": 0.0001
  }
}
```

---

## Configuration

Model settings live in `config.yaml`:

```yaml
output:
  data_dir: "./data"
  response_file: "responses.jsonl"
  append_mode: true

default:
  temperature: 1.0
  max_tokens: 2048
  top_p: 1.0

models:
  gpt-4:
    provider: "openai"
    model_name: "gpt-4"
    api_key: "${OPENAI_API_KEY}"
```

Environment variables can be configured with `.env`:

```bash
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key
HUGGINGFACE_API_KEY=your-huggingface-api-key
```

Never commit a real `.env` file or real API keys.

---

## Data Analysis

```python
stats = generator.get_statistics()
print(f"Total Queries: {stats['total_queries']}")
print(f"Total Tokens: {stats['total_tokens_used']}")
print(f"Avg Latency: {stats['average_time_per_query']:.2f}s")
```

Export to CSV:

```python
import pandas as pd

responses = generator.load_responses()
data = [r.to_dict() for r in responses]
df = pd.DataFrame(data)
df.to_csv("data/responses.csv", index=False)
```

More examples are available in [examples/05_data_analysis.py](examples/05_data_analysis.py).

---

## Notes

1. API key safety: use `.env` or environment variables and never commit real keys.
2. Cost control: track token usage and estimated costs when using paid APIs.
3. Rate limits: add delays in batch mode to avoid throttling.
4. Local models: no API cost, but require sufficient hardware.

---

## License

This module is independent from the main project and can be used and modified freely.

---

## Contributing

Issues and pull requests are welcome.
