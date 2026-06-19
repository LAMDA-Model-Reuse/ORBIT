# Examples Guide

All examples are designed to be **self-contained** and can be run independently. They are numbered from easiest to more advanced.

---

## Quick Navigation

| No.  | File                     | Description                          | Best For                         |
| ---- | ------------------------ | ------------------------------------ | -------------------------------- |
| 01   | `01_quick_start.py`      | Minimal getting-started example      | First run, quick smoke test      |
| 02   | `02_local_model.py`      | Run local HuggingFace models         | Offline use, already have models |
| 03   | `03_batch_processing.py` | Batch data generation                | Large-scale data collection      |
| 04   | `04_model_comparison.py` | Compare and evaluate multiple models | Choosing the best model          |
| 05   | `05_data_analysis.py`    | Analyze and export collected data    | Understanding and exporting data |

---

## How to Run

```bash
cd examples

# Run any example
python 01_quick_start.py
python 02_local_model.py
# ... and so on
```

------

## Details

### 01_quick_start.py

**The simplest way to get started**

- ✅ Uses online APIs (OpenAI / Anthropic / Google)
- ✅ Single-query example
- ✅ Automatically saves results

**Prerequisites**

- Set your API keys in `.env`
- Or update model configuration in `config.yaml`

**Sample Output**

```
Query: What is the capital of France?
Response: The capital of France is Paris.
Tokens: 25 (10 + 15)
Time: 1.5s
```

------

### 02_local_model.py

**Run local models (no API key required)**

Includes four sub-examples:

- **a) Basic usage**: load a model and generate a single response
- **b) Memory optimization**: 8-bit / 4-bit quantization
- **c) Batch generation**: process multiple queries in one run
- **d) Parameter exploration**: compare different temperatures

**Prerequisites**

```bash
pip install transformers torch accelerate
# Optional (for quantization)
pip install bitsandbytes
```

**Sample Output**

```
Loading model: meta-llama/Llama-3.2-1B
✓ Model loaded successfully!

Query: What is the capital of France?
Response: The capital of France is Paris.
Tokens: 42
Time: 0.8s
Speed: 50 tokens/s
```

------

### 03_batch_processing.py

**Batch data generation**

Includes four sub-examples:

- **a) API batch mode**: use online APIs
- **b) Local model batch mode**: use local models
- **c) Large-scale collection**: handle many queries
- **d) Read from file**: load queries from a JSONL file

**Use Cases**

- Building training datasets
- Evaluating model performance
- Collecting many responses efficiently

**Sample Output**

```
Processing 5 queries...
100%|████████████| 5/5 [00:10<00:00, 2.0s/it]

Batch Summary:
Total queries: 5
Success: 5
Failed: 0
Total tokens: 350
Total time: 10.5s
```

------

### 04_model_comparison.py

**Model comparison and evaluation**

Includes four sub-examples:

- **a) API model comparison**: GPT vs Claude vs Gemini
- **b) Local model comparison**: different sizes / quantization setups
- **c) Quality comparison**: compare different temperatures
- **d) Speed comparison**: throughput benchmarking

**Comparison Dimensions**

- Response quality
- Token usage
- Latency / speed
- Cost (for API models)

**Sample Output**

```
Comparing 3 models...

[gpt-3.5-turbo]
Tokens: 120 (15 + 105)
Time: 2.1s
Speed: 50 tokens/s
Cost: $0.000180

[claude-3-haiku]
Tokens: 135 (15 + 120)
Time: 1.8s
Speed: 67 tokens/s
Cost: $0.000203
```

------

### 05_data_analysis.py

**Data analysis and export**

Includes five sub-examples:

- **a) Basic stats**: dataset overview
- **b) Per-model analysis**: detailed metrics by model
- **c) Export to CSV**: convert JSONL into a tabular file
- **d) Filtering**: subset records by conditions
- **e) Length distribution**: analyze output length statistics

**What You Can Analyze**

- Token usage distribution
- Latency and performance
- Success rate
- Model-level comparisons

**Sample Output**

```
Dataset Overview:
  Total queries: 50
  Success: 48
  Failed: 2
  Success rate: 96.0%

Token Usage:
  Total: 3,500
  Avg per query: 72.9

Models Used:
  meta-llama/Llama-3.2-1B: 30 runs
  gpt-3.5-turbo: 20 runs
```

------

## Learning Paths

### 🎯 Beginner Path

1. Run `01_quick_start.py` to learn the basics
2. Run `02_local_model.py` to try local models
3. Run `05_data_analysis.py` to inspect saved data

### 🚀 Advanced Path

1. Collect data at scale with `03_batch_processing.py`
2. Compare models using `04_model_comparison.py`
3. Deep-dive analysis with `05_data_analysis.py`

### 💡 Practical Workflow

1. Collect your dataset: edit the query list in `03_batch_processing.py`
2. Choose the best model: use `04_model_comparison.py`
3. Export and analyze: run `05_data_analysis.py` to export to CSV

------

## Customizing the Examples

All example scripts are easy to modify.

### Switch Models

```python
# Online API
model_name = "gpt-4"  # change to your target model

# Local model
model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
```

### Change Queries

```python
query = "Your question"
# or
queries = ["Question 1", "Question 2", "Question 3"]
```

### Tune Parameters

```python
temperature = 0.7  # 0.1–2.0 controls randomness
max_tokens = 100   # maximum generation length
top_p = 0.9        # nucleus sampling
```

------

## Output Files

All examples write data to:

```text
data/
├── responses.jsonl    # main data file (auto-appended)
├── responses.csv      # CSV export (generated on demand)
└── ...                # other output files
```

------

## FAQ

### Q: An example fails to run—what should I check?

**A:** Verify:

1. Dependencies: `pip install -r ../requirements.txt`
2. API keys (if using online APIs)
3. The error message (it usually points to the cause)

### Q: Which example should I start with?

**A:** Pick based on your goal:

- **Quick smoke test**: 01
- **Local/offline models**: 02
- **Large-scale data**: 03
- **Model selection**: 04
- **Data analysis**: 05

### Q: Can I run multiple examples at the same time?

**A:** Yes. Each example is independent, and results will be appended to the same output file by default.

### Q: How do I start fresh?

**A:** Delete or rename `data/responses.jsonl`.

------

## Need Help?

- 📖 Full docs: `../README.md`
- 🔧 Local models: `../docs/LOCAL_TRANSFORMERS_GUIDE.md`
- 💡 Quickstart: `../docs/QUICKSTART.md`

------

**Tip**: Each example file includes detailed comments at the top—read them before running for the smoothest experience.
