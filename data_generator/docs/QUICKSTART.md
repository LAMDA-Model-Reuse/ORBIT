# Quickstart Guide

## Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
cd data_generator
pip install -r requirements.txt
```

### Step 2: Configure API Keys

Choose one of the following options:

**Option A: Use an environment file (recommended)**

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

**Option B: Edit the config file directly**

```bash
# Edit config.yaml and replace ${API_KEY} with your real key
```

### Step 3: Verify Installation

```bash
python -c "from generator import LLMDataGenerator; print('OK')"
```

### Step 4: Run Your First Example

```bash
python examples/01_quick_start.py
```

---

## Common Commands

### Single Query

```python
from generator import LLMDataGenerator

generator = LLMDataGenerator()
response = generator.generate_single(
    query="Your question",
    model_name="gpt-3.5-turbo"
)
```

### Batch Processing

```bash
python examples/03_batch_processing.py
```

### Data Analysis

```bash
python examples/05_data_analysis.py
```

---

## Supported Models

### Closed-Source Models (API key required)

- **OpenAI**: `gpt-4`, `gpt-3.5-turbo`
- **Anthropic**: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- **Google**: `gemini-pro`, `gemini-2.5-pro`

### Open Models (Run locally)

- **Local deployment**: Llama 3/3.1, Qwen, Mistral, etc.
- Start a server using vLLM or TGI, or use the direct Transformers generator.
- See: [LOCAL_MODEL_GUIDE.md](LOCAL_MODEL_GUIDE.md)

---

## Configuration Overview

Add models in `config.yaml`:

```yaml
models:
  your-model-name:
    provider: "openai"  # or anthropic, google, ollama
    model_name: "gpt-3.5-turbo"
    api_key: "${OPENAI_API_KEY}"
```

---

## Data Storage

All records are saved automatically under `data/`:

- `responses.jsonl`: full response records
- One JSON object per line, including query, response, tokens, timing, and metadata.

---

## Next Steps

- Read the full documentation: [README.md](../README.md)
- Explore more examples: `examples/`
- Customize your setup: edit `config.yaml`

---

## FAQ

**Q: Can I use this without an API key?**
A: Yes. Run open-source models locally via vLLM or TGI, or directly with Transformers. See `LOCAL_MODEL_GUIDE.md`.

**Q: How do I add a new model?**
A: Add a new entry under `models` in `config.yaml`.

**Q: Where is the data saved?**
A: By default: `data/responses.jsonl`.

**Q: How do I export to CSV?**
A: Run `python examples/05_data_analysis.py`.
