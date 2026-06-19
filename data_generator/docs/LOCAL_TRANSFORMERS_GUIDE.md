# Run Local HuggingFace Models Directly (No vLLM Required)

If you’ve already downloaded a model from HuggingFace, you can load and run it **directly in Python** — **no server needed**.

---

## Quick Start

### 1) Install Dependencies

```bash
# Core dependencies
pip install transformers torch accelerate

# Optional: quantization to reduce memory usage
pip install bitsandbytes
```

### 2) Run the Example

```bash
cd data_generator/examples
python local_model_example.py
```

------

## Usage

### Basic Usage

```python
from local_generator import LocalLLMGenerator

# Option 1: Use a HuggingFace model ID (loads from cache or downloads automatically)
generator = LocalLLMGenerator(
    model_path="meta-llama/Meta-Llama-3.2-1B-Instruct",
    device="auto",  # auto-select GPU/CPU
    verbose=True
)

# Option 2: Use a local path (if the model is already downloaded)
generator = LocalLLMGenerator(
    model_path="/path/to/your/Llama-3.1-8B-Instruct",
    device="auto",
    verbose=True
)

# Generate a response
response = generator.generate_single(
    query="What is the capital of France?",
    temperature=0.7,
    max_tokens=100,
    save_result=True
)

print(response.response)
print(f"Tokens: {response.token_usage.total_tokens}")
print(f"Time: {response.timing_info.duration_seconds:.2f}s")
```

### Batch Generation

```python
queries = [
    "What is AI?",
    "What is machine learning?",
    "What is deep learning?"
]

responses = generator.generate_batch(
    queries=queries,
    temperature=0.7,
    max_tokens=100,
    show_progress=True
)
```

------

## Memory Optimization (8-bit / 4-bit Quantization)

If your GPU memory is limited, enable quantization:

```python
# 8-bit quantization (~50% memory savings)
generator = LocalLLMGenerator(
    model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    device="auto",
    load_in_8bit=True,
    verbose=True
)

# 4-bit quantization (~75% memory savings, slightly lower quality)
generator = LocalLLMGenerator(
    model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    device="auto",
    load_in_4bit=True,
    verbose=True
)
```

------

## Finding Your Model Path

### Case 1: Use a HuggingFace Model ID

Just use the model ID — caching and downloading are handled automatically:

```python
model_path = "meta-llama/Meta-Llama-3.2-1B-Instruct"
# or
model_path = "Qwen/Qwen2.5-7B-Instruct"
# or
model_path = "mistralai/Mistral-7B-Instruct-v0.2"
```

### Case 2: Model Stored in a Custom Directory

If you downloaded the model to a specific location:

```python
model_path = "/data/models/Llama-3.1-8B-Instruct"
# or
model_path = "/home/user/Downloads/Qwen2.5-7B"
```

### Case 3: Model in the HuggingFace Cache

HuggingFace caches models under `~/.cache/huggingface/hub/` by default:

```bash
ls ~/.cache/huggingface/hub/

# Examples:
# models--meta-llama--Meta-Llama-3.1-8B-Instruct/
# models--Qwen--Qwen2.5-7B-Instruct/
```

You can use the full snapshot path:

```python
model_path = "~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/abc123..."
```

Or just use the model ID (recommended):

```python
model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
```

------

## Generation Parameters

```python
response = generator.generate_single(
    query="Your question",

    # Sampling
    temperature=0.7,      # 0-2; higher = more random
    top_p=0.9,            # 0-1; nucleus sampling
    top_k=50,             # sample from top-k tokens
    do_sample=True,       # False = greedy decoding, True = sampling

    # Length
    max_tokens=2048,      # max generated tokens

    # Saving
    save_result=True,
    metadata={"task": "qa"}  # extra metadata
)
```

### Temperature Guidelines

- **0.1–0.3**: factual QA, translation, code (precision-focused)
- **0.7–0.9**: general chat, rewriting
- **1.0–1.5**: creative writing, brainstorming
- **1.5+**: highly random (rarely recommended)

------

## Device Configurations

### NVIDIA GPU (CUDA)

```python
generator = LocalLLMGenerator(
    model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    device="cuda",
    torch_dtype="float16",
    verbose=True
)
```

### Apple Silicon (MPS)

```python
generator = LocalLLMGenerator(
    model_path="meta-llama/Meta-Llama-3.2-1B-Instruct",
    device="mps",
    torch_dtype="float16",
    verbose=True
)
```

### CPU Only

```python
generator = LocalLLMGenerator(
    model_path="meta-llama/Meta-Llama-3.2-1B-Instruct",
    device="cpu",
    torch_dtype="float32",
    verbose=True
)
```

### Auto (Recommended)

```python
generator = LocalLLMGenerator(
    model_path="meta-llama/Meta-Llama-3.2-1B-Instruct",
    device="auto",
    torch_dtype="auto",
    verbose=True
)
```

------

## Recommended Models

### Small Models (Good for Testing, 1–3B)

```python
"meta-llama/Meta-Llama-3.2-1B-Instruct"  # fast smoke tests
"meta-llama/Meta-Llama-3.2-3B-Instruct"  # better quality
"Qwen/Qwen2.5-1.5B-Instruct"
```

### Mid-size Models (7–8B, Common Production Choice)

```python
"meta-llama/Meta-Llama-3.1-8B-Instruct"  # strong baseline
"Qwen/Qwen2.5-7B-Instruct"               # strong Chinese performance
"mistralai/Mistral-7B-Instruct-v0.3"
```

### Large Models (70B+, Requires Multi-GPU or Quantization)

```python
"meta-llama/Meta-Llama-3.1-70B-Instruct"
"Qwen/Qwen2.5-72B-Instruct"
```

------

## Memory Requirements (Rule of Thumb)

| Model Size | FP16   | 8-bit | 4-bit  | Suggested Hardware     |
| ---------- | ------ | ----- | ------ | ---------------------- |
| 1B         | ~2GB   | ~1GB  | ~0.6GB | Any GPU / CPU          |
| 3B         | ~6GB   | ~3GB  | ~1.8GB | GTX 1660+              |
| 7–8B       | ~16GB  | ~8GB  | ~4.5GB | RTX 3060 12GB+         |
| 13B        | ~26GB  | ~13GB | ~7GB   | RTX 3090 / 4090        |
| 70B        | ~140GB | ~70GB | ~35GB  | A100 80GB or multi-GPU |

------

## Complete Example

```python
from local_generator import LocalLLMGenerator

# Initialize
generator = LocalLLMGenerator(
    model_path="/data/models/Llama-3.1-8B-Instruct",
    device="auto",
    torch_dtype="float16",
    load_in_8bit=False,
    verbose=True
)

# Single query
response = generator.generate_single(
    query="Explain deep learning in Chinese.",
    temperature=0.7,
    max_tokens=500,
    save_result=True
)

print(f"Answer: {response.response}")
print(f"Time: {response.timing_info.duration_seconds:.2f}s")
print(f"Tokens: {response.token_usage.total_tokens}")

# Batch queries
queries = [
    "What is machine learning?",
    "What is a neural network?",
    "What is deep learning?"
]

responses = generator.generate_batch(
    queries=queries,
    temperature=0.7,
    max_tokens=200,
    save_results=True,
    show_progress=True
)

# Load saved records
all_responses = generator.load_responses()
print(f"Saved {len(all_responses)} responses in total")
```

------

## FAQ

### Q: Where is my model located?

Run:

```bash
# Check the HuggingFace cache
ls ~/.cache/huggingface/hub/

# Or search for model directories (may take a while)
find ~ -name "*llama*" -type d 2>/dev/null | grep models
```

### Q: Out of memory (OOM). What should I do?

Use quantization:

```python
generator = LocalLLMGenerator(
    model_path="your-model",
    load_in_8bit=True,  # or load_in_4bit=True
    device="auto"
)
```

### Q: It’s too slow. Any tips?

1. Ensure you’re using a GPU (`device="cuda"`)
2. Use FP16 (`torch_dtype="float16"`)
3. Reduce `max_tokens`
4. Use a smaller model (1B–3B)

### Q: How do I force a specific GPU?

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use GPU 0
```

Or via CLI:

```bash
CUDA_VISIBLE_DEVICES=0 python local_model_example.py
```

### Q: How does this compare to vLLM?

| Feature  | LocalLLMGenerator (This Approach) | vLLM                         |
| -------- | --------------------------------- | ---------------------------- |
| Setup    | In-code loading                   | Requires a server            |
| Ease     | ⭐⭐⭐⭐⭐ Very easy                   | ⭐⭐⭐ Medium                   |
| Speed    | ⭐⭐⭐ Standard                      | ⭐⭐⭐⭐⭐ High-performance       |
| Batching | ⭐⭐⭐ Mostly sequential             | ⭐⭐⭐⭐⭐ Dynamic batching       |
| Best for | Quick tests, data collection      | Production, high concurrency |

**Recommendation**

- **Quick testing / data collection**: use `LocalLLMGenerator`
- **Production / heavy traffic**: use vLLM

------

## Troubleshooting

### Error: `CUDA out of memory`

```python
# Fix 1: enable quantization
generator = LocalLLMGenerator(
    model_path="your-model",
    load_in_8bit=True
)

# Fix 2: use a smaller model
model_path = "meta-llama/Meta-Llama-3.2-1B-Instruct"

# Fix 3: free memory (if supported by your generator)
generator.clear_memory()
```

### Error: `No module named 'transformers'`

```bash
pip install transformers torch accelerate
```

### Error: `bitsandbytes not found`

```bash
# Required if using load_in_8bit or load_in_4bit
pip install bitsandbytes
```

------

## More Examples

See the full example script:

```bash
python examples/local_model_example.py
```

It includes:

- Basic usage
- Quantization examples
- Batch generation
- Parameter comparisons
- Loading from local paths
