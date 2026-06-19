# Local Llama Model Usage Guide

If you have already downloaded Llama models locally (e.g., via HuggingFace), you can run them using one of the following approaches.

## Option 1: vLLM (Recommended)

vLLM is a high-performance inference engine that exposes an **OpenAI-compatible API**.

### Install vLLM

```bash
pip install vllm
```

### Start a vLLM Server

```bash
# Assume your model is located at /path/to/Llama-3.1-8B-Instruct
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --max-model-len 8192
```

Or use a HuggingFace model ID (vLLM will download it automatically):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

### Configure `config.yaml`

Add the following in `config.yaml`:

```yaml
models:
  my-local-llama:
    provider: "openai"  # vLLM exposes an OpenAI-compatible API
    model_name: "meta-llama/Meta-Llama-3.1-8B-Instruct"  # must match the server model
    api_base: "http://localhost:8000/v1"
    api_key: "EMPTY"  # local servers typically don't require a real API key
```

### Usage

```python
from generator import LLMDataGenerator

generator = LLMDataGenerator(config_path="config.yaml")
response = generator.generate_single(
    query="What is the capital of France?",
    model_name="my-local-llama",
    save_result=True
)
```

---

## Option 2: Text Generation Inference (TGI)

TGI is HuggingFace’s official inference server, and it can also expose an **OpenAI-compatible API**.

### Run TGI (Docker Recommended)

```bash
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v /path/to/models:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Meta-Llama-3.1-8B-Instruct
```

### Configure `config.yaml`

```yaml
models:
  my-local-llama-tgi:
    provider: "openai"  # TGI can provide an OpenAI-compatible API
    model_name: "meta-llama/Meta-Llama-3.1-8B-Instruct"
    api_base: "http://localhost:8080/v1"
    api_key: "EMPTY"
```

---

## Option 3: llama.cpp + llama-cpp-python

A good choice if you don’t have a GPU or want to run **quantized GGUF** models.

### Install

```bash
pip install llama-cpp-python[server]
```

### Start the Server

```bash
# Assume you have a GGUF quantized model
python -m llama_cpp.server \
    --model /path/to/llama-3.1-8b-instruct.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8000 \
    --n_ctx 8192
```

### Configure `config.yaml`

```yaml
models:
  my-local-llama-cpp:
    provider: "openai"
    model_name: "llama-3.1-8b-instruct"
    api_base: "http://localhost:8000/v1"
    api_key: "EMPTY"
```

---

## Option 4: HuggingFace Transformers (Not Recommended for Batch Inference)

If you want to load models directly in Python, you typically need to modify your generator implementation (e.g., `generator.py`). This bypasses LiteLLM’s unified interface and is not ideal for production.

### Create a Custom Generator

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LocalLlamaGenerator:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 1.0):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()  # remove prompt prefix

        input_tokens = len(inputs["input_ids"][0])
        output_tokens = len(outputs[0]) - input_tokens
        return response, input_tokens, output_tokens

# Example usage
generator = LocalLlamaGenerator("/path/to/Llama-3.1-8B-Instruct")
response, input_tokens, output_tokens = generator.generate("What is the capital of France?")
print(f"Response: {response}")
print(f"Tokens: {input_tokens + output_tokens}")
```

> **Note**: This approach does not leverage LiteLLM’s unified interface and is not recommended for production workloads.

---

## Recommended Setup

### GPU Server

* **Best choice**: vLLM (fastest, supports PagedAttention)
* Great for batch inference with automatic batching

### CPU Server / Low Memory

* **Best choice**: llama.cpp + GGUF quantized models (e.g., Q4_K_M or Q5_K_M)
* Slower, but much lower resource usage

### Docker-first Environment

* **Best choice**: TGI
* Officially maintained and generally stable

---

## FAQ

### Q: My model is at `/data/models/Llama-3.1-8B-Instruct`. How do I configure it?

```yaml
# config.yaml
models:
  my-llama:
    provider: "openai"
    model_name: "/data/models/Llama-3.1-8B-Instruct"  # local path
    api_base: "http://localhost:8000/v1"
    api_key: "EMPTY"
```

Then start vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /data/models/Llama-3.1-8B-Instruct \
    --port 8000
```

### Q: How can I verify the server is running?

```bash
curl http://localhost:8000/v1/models
```

You should see a list of available models.

### Q: Can I run multiple models at the same time?

Yes—use different ports:

```bash
# Terminal 1: Llama-3.1-8B on port 8000
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1

# Terminal 2: Qwen-2.5-7B on port 8001
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8001 \
    --tensor-parallel-size 1
```

And configure:

```yaml
models:
  my-llama:
    provider: "openai"
    model_name: "meta-llama/Meta-Llama-3.1-8B-Instruct"
    api_base: "http://localhost:8000/v1"
    api_key: "EMPTY"

  my-qwen:
    provider: "openai"
    model_name: "Qwen/Qwen2.5-7B-Instruct"
    api_base: "http://localhost:8001/v1"
    api_key: "EMPTY"
```

---

## vLLM Performance Tuning

```bash
# High-throughput config (good for A100/H100)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1 \        # single GPU
    --gpu-memory-utilization 0.9 \    # target GPU memory usage
    --max-num-seqs 256 \              # max concurrent sequences
    --max-model-len 8192 \            # context length
    --dtype auto                      # choose dtype automatically
```

```bash
# Multi-GPU example (e.g., 2x A100)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2 \        # tensor parallel across 2 GPUs
    --gpu-memory-utilization 0.95
```

---

## Getting Started

1. Pick an option (vLLM recommended)
2. Start the local server
3. Add your model to `config.yaml`
4. Run `examples/01_quick_start.py` to verify everything works

```bash
cd data_generator/examples
python quick_start.py
```
