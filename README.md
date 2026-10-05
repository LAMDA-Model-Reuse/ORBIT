<p align="center">
  <img src="./resources/logo.jpg" width="600px" alt="ORBIT logo">
</p>

<p align="center">
  <a href="#introduction">Introduction</a> ·
  <a href="#orbit-and-routejudge">RouteJudge</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#methods-reproduced">Methods</a> ·
  <a href="#evaluation-metrics">Metrics</a> ·
  <a href="#benchmark-suite">Benchmarks</a> ·
  <a href="#extending-orbit">Extending</a> ·
  <a href="#contribution-guidelines">Contributing</a> ·
  <a href="#citing-orbit-and-routejudge">Citation</a> ·
  <a href="#contact">Contact</a>
</p>
## Introduction

**ORBIT (Optimal Routing and Budgeted Inference Toolbox)** is a modular, extensible toolbox for **LLM routing**. It studies how to select the most suitable model from a heterogeneous model pool **for each query**, under practical deployment constraints such as **cost, latency, and throughput**.

<p align="center">
  <img src="./resources/orbit.jpg" width="600px" alt="ORBIT overview">
</p>

LLM routing methods are rapidly emerging, but existing implementations are often fragmented: they use different benchmark splits, cost assumptions, evaluation scripts, and router interfaces. This makes fair comparison and reproducibility difficult.

ORBIT standardizes the end-to-end routing workflow into a unified research stack:

- **Unified pipeline and router interface**: dataset loading -> embedding extraction -> router training/inference -> standardized evaluation, all driven by consistent JSON configs.
- **Budget-aware evaluation**: sweep budgets to trace performance-cost trade-offs and report curve-level metrics such as **nAUC**, **Peak Score**, **QNC**, and **RCI**.
- **Extensible benchmark and method suite**: built-in support for unimodal and multimodal routing benchmarks, reproduced routing methods, and clean interfaces for adding datasets, embedding encoders, and routers.

ORBIT currently supports large-scale routing evaluation with up to **8.65M** benchmark instances and is designed to accelerate reproducible research on budget-aware LLM routing systems.

---

## ORBIT and RouteJudge

ORBIT is the standardized development and integration layer for **RouteJudge**, an open platform for reproducible and preference-aware LLM routing evaluation.

While ORBIT focuses on offline router development, benchmarking, and reproducible protocol design, RouteJudge extends routing evaluation to **online user preference feedback**. In RouteJudge, multiple routers recommend models under the same model pool and budget constraints; selected responses are shown to users through anonymous pairwise comparison; and user preferences are attributed back to the routers behind the compared responses.

Researchers can use ORBIT to implement and validate new routers, then submit compatible methods for RouteJudge historical replay and online preference-based evaluation.

We welcome contributions from the community:

- Submit a pull request to add a new router, benchmark, embedding encoder, or evaluation utility.
- Open an issue if you would like feedback before implementing a larger method.
- Email us if you would like to integrate your routing method into ORBIT and RouteJudge evaluation.

Useful links:

- RouteJudge platform: [https://routejudge.cn](https://routejudge.cn)
- RouteJudge paper: [arXiv:2606.18774](https://arxiv.org/abs/2606.18774)
- ORBIT repository: [https://github.com/LAMDA-Model-Reuse/ORBIT](https://github.com/LAMDA-Model-Reuse/ORBIT)

---

## What's New

- **2026-10**: Added **RouteFM**, the frozen in-context foundation router that transfers to new candidate pools through behavioral context, and **SAVERouter**, which adaptively acquires fixed-K sparse feedback and accounts for supervision expenditure. [[RouteFM Paper]](https://arxiv.org/abs/2609.37362) [[RouteFM Code]](https://github.com/LAMDA-Model-Reuse/RouteFM) [[SAVERouter Paper]](https://arxiv.org/abs/2609.37402) [[SAVERouter Code]](https://github.com/LAMDA-Model-Reuse/SaveRouter)
- **2026-08**: ORBIT has been accepted by the *Frontiers of Computer Science* (FCS) special column **[Code & Data](https://journal.hep.com.cn/fcs/EN/subject/showCollection.do?subjectId=1710741206314)**. [[Paper]](https://doi.org/10.1007/s11704-026-61310-5)
- **2026-08**: Added InferenceDynamics.
- **2026-08**: Added the LLMRouterBench Performance-Cost benchmark with 12 models across 10 tasks, using ORBIT's standardized split and globally scaled per-query costs.
- **2026-06**: Initial release of ORBIT v1.0 with a unified routing pipeline, standardized budget-aware evaluation, unimodal and multimodal benchmarks, and reproduced routing methods.

---

## Methods Reproduced

ORBIT reproduces **30 representative LLM routing methods** across training-free, retrieval-based, and learned routers under a unified pipeline and standardized budgeted evaluation.

- **Avengers**: A training-free recipe that clusters queries and routes by cluster-wise capability profiles with sampling/voting. [[Paper]](https://arxiv.org/abs/2505.19797)
- **Avengers-Pro**: A test-time routing framework that traces a Pareto frontier via clustering and a tunable performance-efficiency objective. [[Paper]](https://arxiv.org/abs/2508.12631)
- **EmbedLLM**: Learns compact vector representations of LLMs to enable efficient performance prediction and scalable routing over many candidates. [[Paper]](https://arxiv.org/abs/2410.02223)
- **Eagle**: A training-free router that maintains global/local Elo-style rankings to select models efficiently. [[Paper]](https://arxiv.org/abs/2409.15518)
- **EquiRouter**: A decision-aware learning-to-rank router that directly supervises instance-wise model rankings to improve accuracy-cost trade-offs. [[Paper]](https://arxiv.org/abs/2602.03478)
- **GraphRouter**: Builds a heterogeneous task-query-LLM graph and casts routing as inductive edge prediction with GNNs for better generalization. [[Paper]](https://arxiv.org/abs/2410.03834)
- **HybridLLM**: Predicts query difficulty and routes between a small and a large model under a tunable desired-quality target. [[Paper]](https://arxiv.org/abs/2404.14618)
- **kNN**: A nonparametric baseline that selects models via nearest neighbors in embedding space using historical per-model outcomes. [[Paper]](https://arxiv.org/abs/2408.12320)
- **MLPRouter**: Uses separate multi-output MLP regressors for query-level performance and cost prediction. [[Paper]](https://arxiv.org/abs/2408.12320)
- **SVMRouter**: Uses per-model support-vector regressors for query-level performance and cost prediction. [[Paper]](https://arxiv.org/abs/2408.12320)
- **MIRT**: An IRT-based router using multidimensional item response theory to jointly estimate model abilities and query attributes. [[Paper]](https://aclanthology.org/2025.acl-long.761/)
- **NIRT**: A neural IRT variant that replaces hand-designed interactions with a neural function for richer ability-difficulty modeling. [[Paper]](https://aclanthology.org/2025.acl-long.761/)
- **ModelSAT**: Learns routing with explicit model capability representations via a capability encoder and a lightweight LLM. [[Paper]](https://arxiv.org/abs/2502.17282)
- **OmniRouter**: Formulates routing as constrained optimization to minimize total cost while satisfying a target performance constraint. [[Paper]](https://arxiv.org/abs/2502.20576)
- **RM-Classification**: A regret-minimization router trained from observational bandit logs using a classification-style surrogate objective. [[Paper]](https://arxiv.org/abs/2505.16037)
- **RM-Interval**: An interval-conditioned regret-minimization router designed to generalize across unseen budget levels by conditioning on cost preferences. [[Paper]](https://arxiv.org/abs/2505.16037)
- **RM-Softmax**: An end-to-end regret-minimization router optimizing a softmax-weighted regret surrogate. [[Paper]](https://arxiv.org/abs/2505.16037)
- **RouteLLM-BERT**: A BERT-based classifier trained on pairwise preference data for routing. [[Paper]](https://arxiv.org/abs/2406.18665)
- **RouteLLM-MF**: Learns low-rank query-model interactions via matrix factorization over preference outcomes for efficient scoring. [[Paper]](https://arxiv.org/abs/2406.18665)
- **RouteLLM-SWRanking**: A similarity-weighted ranking approach that upweights comparisons from prompts similar to the current query. [[Paper]](https://arxiv.org/abs/2406.18665)
- **RouterDC**: Learns query and model embeddings via dual contrastive objectives to better model query-model compatibility. [[Paper]](https://arxiv.org/abs/2409.19886)
- **Oracle**: A non-deployable upper bound that selects the best feasible model per query using ground-truth outcomes.
- **TRouter**: Learns task-aware query and model representations for performance-cost routing. [[Paper]](https://arxiv.org/abs/2604.09377)
- **UniRoute**: Represents model capabilities through cluster-level prediction errors and learns a query-to-cluster router. [[Paper]](https://openreview.net/forum?id=ka82fvJ5f1)
- **InferenceDynamics**: Builds parameter-free model indexes from ranked capability and knowledge profiles for structured model-query matching. [[Paper]](https://arxiv.org/abs/2505.16303)
- **ProfileRouter**: Implements RouteProfile's training-free Emb-GNN profile construction and cosine-similarity SimRouter using public model descriptions, family metadata, and optional reported benchmark scores. [[Paper]](https://arxiv.org/abs/2605.00180)
- **CarrotRouter**: Uses CARROT's KNN plug-in estimators for query-level performance and cost, then applies its original cost-aware rate-optimal scalarization. [[Paper]](https://arxiv.org/abs/2502.03261)
- **EARAMRouter**: Trains independent provider-side success predictors and allocates queries through EA-RAM's largest-positive-surplus reverse auction with runner-up externality payments. [[Paper]](https://arxiv.org/abs/2608.12719)
- **RouteFM**: Uses the official frozen foundation router and anonymous per-candidate behavioral context to predict target-query quality and relative cost without target-domain parameter updates. ORBIT defaults to the released BGE text checkpoint. [[Paper]](https://arxiv.org/abs/2609.37362) [[Code]](https://github.com/LAMDA-Model-Reuse/RouteFM)
- **SAVERouter**: Adaptively reveals exactly K model outcomes per training query, then combines hierarchical group-model shrinkage with query-level residual refinement and native sparse cost estimates. [[Paper]](https://arxiv.org/abs/2609.37402) [[Code]](https://github.com/LAMDA-Model-Reuse/SaveRouter)

---

## Evaluation Metrics

ORBIT evaluates LLM routing under **budgeted inference** by sweeping budgets to obtain a **performance-cost trade-off curve**. It reports standardized metrics for fair comparison and diagnostic analysis.

### Trade-off metrics

- **nAUC (normalized Area Under Curve)**: integrates the Pareto performance envelope over one benchmark-wide realized-cost interval, from the mean per-query minimum model cost to the mean per-query maximum model cost. Every router uses the same real minimum-cost policy as the left anchor, and the final feasible policy is carried to the upper bound. The area is divided by the shared cost interval, so router scores are directly comparable within a benchmark. Higher is better.
- **Peak Score**: reports the best achievable utility within the evaluated budget range. Higher is better.
- **QNC (Quality Neutral Cost)**: measures cost-efficiency under quality-neutral comparison. Lower is better.

### Diagnostic metrics

- **RCI (Routing Collapse Index)**: diagnoses collapse and other failure modes beyond average utility.

### Cost prediction protocol

Every score-based learned router supplies a query-by-model cost matrix without using test labels.
Methods with a native cost mechanism reuse it: for example, cluster statistics for Avengers,
neighbor retrieval for Eagle, cluster features for UniRoute, and profile indexes for
InferenceDynamics. RouteFM uses its released relative-cost head, while SAVERouter estimates
costs from its sparse group/model observations. Methods whose native score cannot represent cost (HybridLLM, RouterDC,
RouteLLM, ModelSAT, ProfileRouter, and EA-RAM's offline execution-cost bids) use the same
multi-output MLP trained only on training-query embeddings
and costs. Its optional settings live under `cost_prediction` (`hidden_sizes`, `dropout`,
`epochs`, `batch_size`, `lr`, and `weight_decay`). Regret-minimization routers instead learn the
performance-cost utility directly as a function of lambda; Oracle intentionally uses realized
test costs as an upper bound.

---

## Quick Start

### Clone and setup

```bash
git clone https://github.com/LAMDA-Model-Reuse/ORBIT
cd ORBIT
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS

pip install -U pip
pip install -r requirement.txt
```

> Note: The commands above are tested on Linux. Windows users may need to resolve library compatibility issues such as `setuptools`, `triton`, or `vllm` according to their local environment.

### Download embedding models

ORBIT supports plugging in any HuggingFace embedding model. We recommend downloading required checkpoints in advance.

```bash
conda install -c conda-forge aria2
curl -L https://hf-mirror.com/hfd/hfd.sh -o hfd.sh
chmod +x hfd.sh
./hfd.sh <HF_MODEL_ID>
```

### Configuration

ORBIT is driven by two JSON configs: one for the benchmark setting and one for the routing algorithm.

- **Dataset-level config** (`configs/benchmarks/[DATASET].json`): defines modality, data protocol, train/test split, query representation, encoder settings, random seeds, logging, and output paths.
- **Method-level config** (`configs/routers/[METHOD].json`): defines router hyperparameters, training setup, and method-specific options.

### Run an experiment

```bash
python main.py --dataset <dataset_name> --method <method_name>
```

Examples:

```bash
python main.py --dataset Routerbench --method AvengersPro
python main.py --dataset MMRBench --method EquiRouter
python main.py --dataset Mixinstruct --method oracle
python main.py --dataset LLMRouterBench --method knn
python main.py --dataset Routerbench --method RouteFM
python main.py --dataset Routerbench --method SaveRouter
```

Dataset and method names are matched case-insensitively. Device selection defaults to
`auto`; use `--device cpu`, `--device cuda`, or `--device cuda:N` to override it.

RouteFM uses the official immutable frozen-router release. Its default BGE adapter expects
`BAAI/bge-base-en-v1.5` at `./bge-base-en-v1.5`; the RouteFM router checkpoint itself is
downloaded and checksum-verified by the official package, or can be supplied through
`routefm.checkpoint` in `configs/routers/RouteFM.json`. SAVERouter uses the paper's per-benchmark
profiles and acquires only the configured fixed-K feedback before fitting.

---

## Benchmark Suite

ORBIT includes a compact benchmark suite spanning text-only and multimodal routing settings, with model pools ranging from small curated sets to large-scale collections.

| Benchmark          | Modality   | #Models | Notes                                                        |
| ------------------ | ---------- | ------: | ------------------------------------------------------------ |
| **RouterBench**    | Text       |      11 | Small text-only pool for controlled budgeted routing evaluation. |
| **RouterEval**     | Text       | Dynamic | Large text-only pool determined by the common models in the downloaded release. |
| **MMR-Bench**      | Multimodal |      10 | Multimodal routing with visual inputs.                       |
| **MixInstruct**    | Text       |      12 | Text-only instruction-style queries across mixed sources.    |
| **LLMRouterBench** | Text       |      12 | Performance-cost routing over ten tasks with per-query model costs. |

> Note: RouterEval provides 12 dataset files with nested model pools. To ensure a consistent evaluation protocol, ORBIT deterministically sorts the common models from Group A's six files and aligns performance and cost columns to that canonical order. The bundled metadata currently contains 239 model descriptions; description-based routers validate coverage by model name and report any missing entries instead of silently misaligning them.

> Note: The LLMRouterBench loader uses the Performance-Cost subset and ORBIT's standard 20/80 in-domain split. It aligns 12 models over 12,446 queries, preserves released costs for auditing, repairs usable zero-cost records from token usage or model-task medians, and globally scales effective costs to `[0, 1]`. Failed zero-usage calls retain zero performance and zero cost.

---

## Extending ORBIT

ORBIT is designed for community extension. You can add new benchmarks, embedding encoders, and routing algorithms while reusing the same data loading, embedding, evaluation, and logging pipeline.

### 1. Adding a new dataset

1. **Generate the dataset in ORBIT format** using the `data_generator` workflow. Specify the model pool, upload your data, and generate the corresponding dataset files.
2. **Register the dataset** in `utils/data.py` and implement the required dataset base class so that it can be loaded by ORBIT's unified pipeline.
3. **Create benchmark configs** under `configs/benchmarks/` to define modality, split protocol, encoder settings, and dataset-level options.
4. **Provide model descriptions** under `configs/description/` if your target methods require them, such as GraphRouter or MIRT.

### 2. Adding a new embedding encoder

1. Register the embedding model in `utils/embedding.py`.
2. Select it in the corresponding benchmark config under `configs/benchmarks/`.

### 3. Adding a new routing algorithm

1. Create a router class under `methods/`. ORBIT provides a `BaseRouter` that already implements dataset loading, embedding extraction, and standardized evaluation.
2. Inherit from `BaseRouter` and implement training plus `predict`, returning two `(N, M)` matrices:
   predicted performance and predicted cost. If the method has no defensible cost estimator, call
   `_fit_shared_cost_predictor` during training and `_predict_shared_cost` during inference.
3. Register the new router in `methods/__init__.py` and `train.py`.
4. Add a JSON config under `configs/routers/` for method hyperparameters.
5. Run offline evaluation and include clear reproduction instructions in your pull request.

If you are unsure how to adapt your method to the ORBIT interface, please open an issue or email us. We are happy to discuss integration and RouteJudge evaluation for new routing methods.

---

## Contribution Guidelines

We welcome pull requests that improve ORBIT's method coverage, benchmark support, documentation, and evaluation utilities.

Before submitting a PR, please try to include:

- A concise description of the method, dataset, or utility being added.
- Configuration files needed to reproduce the experiment.
- Required dependencies or checkpoint instructions.
- Random seeds and preprocessing details when applicable.
- A minimal command-line example.

For larger additions, especially new routing methods intended for RouteJudge evaluation, please open an issue or email us first so that we can coordinate the expected interface and evaluation protocol.

---

## Citing ORBIT and RouteJudge

If you use ORBIT or RouteJudge in your research, please cite:

### ORBIT

Guannan Lai, Haoran Hu, Hao-Xuan Ma, Han-Jia Ye. **ORBIT: An Optimal Routing and Budgeted Inference Toolbox.** *Frontiers of Computer Science*, 2026. DOI: [10.1007/s11704-026-61310-5](https://doi.org/10.1007/s11704-026-61310-5)

```bibtex
@article{lai2026orbit,
  title   = {{ORBIT}: An Optimal Routing and Budgeted Inference Toolbox},
  author  = {Lai, Guannan and Hu, Haoran and Ma, Hao-Xuan and Ye, Han-Jia},
  journal = {Frontiers of Computer Science},
  year    = {2026},
  doi     = {10.1007/s11704-026-61310-5},
  url     = {https://doi.org/10.1007/s11704-026-61310-5}
}
```

### RouteJudge

```bibtex
@inproceedings{lai2026routejudge,
  title     = {RouteJudge: Preference-Based Evaluation of {LLM} Routers under Pluralistic User Preferences},
  author    = {Guannan Lai and Haoran Hu and Han-Jia Ye},
  booktitle = {Pluralistic Alignment Workshop at ICML 2026},
  year      = {2026}
}

@misc{lai2026routejudgeopenplatformreproducible,
  title         = {RouteJudge: An Open Platform for Reproducible and Preference-Aware LLM Routing},
  author        = {Guannan Lai and Haoran Hu and Han-Jia Ye},
  year          = {2026},
  eprint        = {2606.18774},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2606.18774}
}
```

---

## Acknowledgments

We thank the authors of the following open-source repositories for their contributions:

- [GraphRouter](https://github.com/ulab-uiuc/GraphRouter)
- [routerbench](https://github.com/withmartian/routerbench)
- [RouterEval](https://github.com/MilkThink-Lab/RouterEval)
- [MMR-Bench](https://github.com/Hunter-Wrynn/MMR-Bench)
- [LLMRouterBench](https://github.com/ynulihao/LLMRouterBench)
- [OmniRouter](https://github.com/dongyuanjushi/OmniRouter)

---

## Contact

For questions, feature requests, or contributions:

- Open an issue on [GitHub](https://github.com/LAMDA-Model-Reuse/ORBIT/issues).
- Email the authors at [laign@lamda.nju.edu.cn](mailto:laign@lamda.nju.edu.cn).

If you would like to add your routing method to ORBIT or have it considered for RouteJudge evaluation, please contact us by issue or email.

