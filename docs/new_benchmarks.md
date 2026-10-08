# New benchmark adapters

All three adapters are registered `BaseDatasetLoader` subclasses. They return
the existing `(train_df, test_df, model_list)` tuple with `prompt`, `eval_name`,
`model_i_performance` and `model_i_cost` columns. No router, embedding interface,
training entry point or metric implementation is changed. Candidate descriptions
are keyed by the same names as `model_list`, including R2's budget actions.
Task-description JSONs are provided for TRouter as well; they describe released
task names, never sample answers. Other xRoute scenarios use TRouter's existing
task-name fallback. R2 has a single released task group, so task-conditioned
methods cannot exploit a finer taxonomy without separately supplied metadata.

```bash
python main.py --dataset XRouteBench --method knn --device cpu
python main.py --dataset MMRBenchV2 --method GraphRouter
python main.py --dataset R2Bench --method CarrotRouter --device cpu
```

These are normal benchmark runs, not lightweight smoke commands. Install the
existing requirements and prepare the configured embedding checkpoints first.
MMR-Bench V2 additionally needs upstream image assets; the complete R2 release
is approximately 25 GB. Select smaller pools through the benchmark JSON when
storage, memory or training time is limited.

## Shared protocol

- Hugging Face revisions are frozen in `configs/benchmarks/*.json`. Only requested
  shards are downloaded into `<dataset_dir>/hub`; archives and remote pickles are
  not used by these adapters.
- Files already exported under `dataset_dir` take precedence, using the original
  Hugging Face layout. Set `dataset.local_files_only=true` to require those files
  and prevent network access. Such inputs are explicitly marked as local
  overrides whose release revision has not been independently verified.
- `dataset.models` selects a nonempty list of distinct source model IDs. The
  default pools are fixed by released metadata, never selected using test scores.
  Model order and all columns use the same deterministic canonical identities.
- Duplicates/conflicting identities, invalid scores, negative/nonfinite costs and
  token counts fail explicitly. Missing labels are never filled with zero.
  `missing_policy=drop-query` retains only complete common candidate support;
  `missing_policy=error` rejects incomplete support. The first is the default for
  MMR V2/R2, the second for xRouteBench. Coverage exclusions are reported by task.
- xRouteBench defaults to its official train/test partitions. MMR V2 and R2 have
  no published router train/test partition in the loaded artifacts: they use
  ORBIT's seeded 20/80 split, grouping identical router inputs together. These are
  **ORBIT protocols**, not claims to reproduce paper split-specific scores.
- `split.mode=in-domain` with `ratios` enables the same grouped random split for
  any adapter. `split.mode=out-of-domain` requires an explicit `split.test_tasks`
  list. R2 CSVs have no released task labels, so they use `eval_name=r2bench` and
  do not support a meaningful task-level OOD partition without external metadata.
- Optional `max_samples` caps the number of input groups, sampled with the seed
  after checking common support (per partition for official/OOD splits). It does
  **not** reduce the size of downloaded source files. It is a subset experiment,
  not the full benchmark.
- `cost_scaling=train-max` divides every model cost by **one training-only maximum**,
  preserving relative costs across models and queries. Test values may exceed 1.
  No row-wise normalization or test-based scale fitting is used. `cost_scaling=none`
  keeps raw units. `model_i_raw_cost` always preserves unscaled reference values.
- Every run logs and saves a content-addressed protocol JSON under
  `<dataset_dir>/orbit_protocols/`, including source revision/overrides, model order,
  cost scope/scale, coverage, split counts and hashes of train/test sample IDs.
  The frames also retain this information in their `attrs`. Keep the sidecar with
  reported results; ORBIT's existing trade-off output format is unchanged.
- Reference answers and recorded responses never become `prompt`. Responses are
  retained for xRouteBench, and optionally for MMR/R2 through
  `include_responses=true` (useful for EA-RAM's optional ex-post evaluator). The
  latter can substantially increase memory usage on the large R2 release.

## xRouteBench

Source: [dataset](https://huggingface.co/datasets/ulab-ai/xRouteBench),
[official code](https://github.com/ulab-uiuc/LLMRouter).

The default `scenario=llmrouter_generic` contains 18 candidates and 3,729 test
queries. In the pinned Parquet files, the 80,802 training rows group into 4,489
`embedding_id` items, not the 4,487 stated in the card; one training prompt is
repeated. `task_id` is absent on many rows and is not a safe join key. The adapter
uses `(scenario, source_split, embedding_id)` and validates that all candidates
agree on the prompt and task. Identical input text cannot cross train/test.

Costs use recorded input/output token counts and the frozen `llm_candidates`
prices, divided by 1,000,000. They are reproducible reference costs, not verified
historical API bills. Model descriptions come from the same candidate artifact.

Other supported `scenario` values are `memory_locomo`, `memory_longmemeval`,
`timeseries`, `video`, `multimodal_geometry3k`, and `multimodal_mathvista`.
All use the released `query` **text**: memory includes the upstream RAG context;
vision/video queries use upstream textual descriptions, not raw image/video
embeddings. This is explicit text-view routing on recorded outcomes.
`personalized` is rejected because its released schema contains pairwise model
preferences, not the required dense performance/cost matrix.

Offline layout:

```text
data/xroutebench/
  llm_candidates/train.parquet
  llmrouter_generic/train.parquet
  llmrouter_generic/test.parquet
```

## MMR-Bench V2

Source: [dataset and asset instructions](https://huggingface.co/datasets/gh0stHunter/MMR-Bench-V2),
[official code](https://github.com/Hunter-Wrynn/MMR-Bench).
The old `MMRBench` adapter and configuration remain unchanged.

The default covers the 44 released model IDs and 18 task sets. Select task sets
using `dataset.benchmarks`, for example `["ERQA", "MMStar"]`; select models using
their exact released IDs, for example
`["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct"]`.

Instance/outcome tables are joined by `sample_id`, with benchmark and model ID
cross-checks. Only `status=ok` outcomes with finite score/cost enter common-support
evaluation. Execution failures are not scored as incorrect answers. Consequently,
this evaluates routing **conditional on successful common support**, not
production failure robustness or the cost of all attempted calls. Sidecars report
the excluded query counts and failure statuses; do not compare results from
different selected pools/support without reporting those differences.

Version 2.0.3 costs are frozen USD **recorded-output reference estimates**. They
exclude unavailable input/image and hidden reasoning usage; 29 of 44 model prices
are low-confidence estimates. The adapter preserves released costs, including
valid zeros, without imputation. Scores are already normalized to `[0,1]`.

`prompt_text` is the router's text input. Image references are relative to
`dataset.image_root` (default `./data/mmrbench_v2/LMUData`); prepare them following
the source dataset card and upstream licensing terms. Missing files and unsafe
paths fail explicitly: there is no automatic text-only fallback.

The existing image embedder takes one path per query. Single-image inputs use
their original path; multi-image inputs use an ordered row-major contact sheet,
preserving every referenced image. Cells default to 512 pixels, with aspect-ratio
preserving downscaling and white padding; `image_cell_size` is configurable from
16 to 2048. Content-addressed PNGs are cached under `contact_sheets/`. This is a
documented routing-feature adaptation, not an identical representation to the
original multi-image model prompts. Raw ordered references remain in `image_refs`.
For an intentional text-only experiment, set `modality=text` and a text-only
embedding configuration; image files are then not required.

Offline layout:

```text
data/mmrbench_v2/
  data/instances/ERQA.parquet
  data/results/Qwen2.5-VL-3B-Instruct/ERQA.parquet
  data/results/Qwen2.5-VL-7B-Instruct/ERQA.parquet
  LMUData/images/ERQA/0.png
  ...
```

## R2-Bench

Source: [CSV release](https://huggingface.co/datasets/JiaqiXue/R2-Bench),
[paper](https://arxiv.org/abs/2602.02823),
[official code](https://github.com/UCF-ML-Research/R2-Router).

This adapter deliberately loads the original **R2-Bench**, not the separate
RouterArena-derived release (RouterArena's official evaluation-only rules must
not be bypassed). The frozen CSV release contains 10 physical models and 157
released model-budget files: the 8,000 file is absent for Qwen2.5-Math-1.5B and
both GLM models. The catalogue in `configs/datasets/r2bench.json` records actual
file availability. It does not invent missing actions or silently replace them.
Explicitly requesting an unavailable model-budget combination fails.

To avoid changing the router interface, each action has an ordinary candidate ID:

```text
Qwen/Qwen3-0.6B@budget=10
Qwen/Qwen3-0.6B@budget=100
...
```

All existing routers can learn/score these columns as usual. Choose an identical
action pool for every compared method. This is **discrete action routing**;
it neither implements R2-Router nor adds continuous-budget interpolation.
Best Single and the existing oracle are computed over candidate **actions**,
not collapsed physical models. The sidecar maps every candidate to its physical
model and nominal budget.

The row join uses the released `key`, with prompt-conflict checks; all budgets
for a query become columns of **one row before splitting**. Input is exclusively
`original_prompt`, never the budget-conditioned `templated_prompt`, reference
answer or model response. Identical original prompts stay in the same partition.
Some model files strip line breaks from otherwise identical prompt content. The
adapter allows **CR/LF-only** differences for the same key, preferring an actually
released original prompt with the most line breaks and recording variant counts.
Other content differences fail; there is no fuzzy matching, answer-based repair,
or budget-template substitution. Consequently, formatting differences in source
generation prompts remain a limitation of cross-model outcome comparison.
Budget names are nominal release labels: some source prompts constrain words,
and actual token counts can exceed the nominal label. Do not claim a hard token
cap or equate budget names with realized cost.

Default `cost_mode=output-usd` uses **actual_token_count** times the frozen output
reference price. Eight prices follow the paper's Table 6; the other two follow
the author's frozen public pricing artifact, with sources recorded in the
registry. These are output-only reference estimates, **not full bills or a claim
to reproduce the paper's full input-inclusive cost protocol**. Missing input-token
and hidden reasoning usage is not guessed. `cost_mode=output-tokens` instead
evaluates output-token consumption; it must not be presented as dollar savings.

For a smaller standard experiment, set:

```json
"models": ["Qwen/Qwen2.5-Math-1.5B-Instruct", "Qwen/Qwen3-0.6B"],
"budgets": [10, 100]
```

With no `budgets` restriction, all actually released actions for selected models
are loaded. A different data revision requires a matching `registry_path` and
description file; pool/price changes must remain explicit and auditable.

Offline layout:

```text
data/r2bench/
  data/Qwen/Qwen3-0.6B/10_judge.csv
  data/Qwen/Qwen3-0.6B/100_judge.csv
  ...
```

## Verification

```bash
python -m unittest discover -s tests -p 'test_new_benchmarks.py' -v
python scripts/smoke_new_benchmarks.py --real-data
```

The regression suite uses offline fixtures matching released schemas and checks
existing retrieval, regression, description, graph, task-conditioned and sparse-supervision routers on all three
adapters. The real-data smoke downloads small xRoute/MMR shards and streams a
bounded R2 CSV prefix; it uses deterministic hash features instead of large
embedding checkpoints and only one training epoch. MMR's real-data smoke is
explicitly text-only; multi-image loading is covered by the offline image tests.
These runs validate plumbing/finite predictions/metrics, **not paper accuracy**
and not an exhaustive run of all methods on all full datasets.

### All-method runtime validation

`tests/test_all_methods_smoke.py` adds a CPU-only matrix covering every registered
method on all three source-schema fixtures, plus all 24 compatible multimodal
methods on MMR image fixtures. It calls real training, prediction, allocation,
nAUC, Peak Score and JSON output code; none of these are mocked. CI discovers this test
through the existing unittest entry point.

For a persistent report with per-method logs and curves:

```bash
# Offline, portable: tiny untrained official RouteFM architecture and automatic
# RouteLLM endpoints; no external annotation calls or pretrained downloads.
python scripts/smoke_all_methods.py --tiny-routefm \
  --report-dir /tmp/orbit-all-methods-offline

# Bounded real benchmark records and an existing released local RouteFM weight.
python scripts/smoke_all_methods.py --real-data \
  --routefm-checkpoint /path/to/routefm_bge.pt \
  --report-dir /tmp/orbit-all-methods-real
```

Report directories must be new or empty to preserve earlier evidence. The report
records substitutions, sample/model counts, checkpoint hashes, nAUC and output
paths. Each method must produce a finite curve and log both nAUC and Peak Score. Pair
routers' intentionally disabled `-inf` scores are accepted only outside the
selected pair. Native cost regressions are checked before and after ORBIT's
existing training-range clipping guard; negative raw values are recorded, not
misrepresented as negative evaluated bills.

The local 2026-10-08 run passed **90/90** real-data method/benchmark combinations:
xRoute had 32 train/32 test queries and 18 candidates; MMR had 6/26 and 2
candidates; R2 had 6/26 and 4 actions. MMR real-record runs were text-only.
Another **24/24** compatible MMR multimodal fixture combinations passed, and the
re-run automatic-pair offline matrix passed **90/90** after the fix below.
These are runtime checks only.

All runs use input-only hash features and one epoch. BERT and ModelSAT execute
real tiny locally initialized Transformer models, including ModelSAT's LoRA,
not their production pretrained checkpoints. NIRT executes UMAP/HDBSCAN and its
router normally but replaces the paid auxiliary-LLM annotation boundary with a
fixed offline response. Real-data RouteFM checks load the existing local BGE
router weights, but use dimension-matched hash features, not real BGE embeddings.
The portable CI fixture instead uses untrained tiny RouteFM weights. Full-data,
production-checkpoint, GPU and paid-service runs are **not** established here.

The RouteLLM SW/MF/BERT automatic-selection boundary found in this validation
has been fixed: inferred endpoints exclude the other candidate, with canonical
index order breaking training-mean ties deterministically. Explicit endpoints
are preserved; an explicitly identical pair, invalid indices or a single-model
pool still fail clearly. The regression matrix now uses automatic endpoints,
including equal-mean fixture pools. This does not fabricate preferences: a
`tie_policy=drop` run with no non-tied labels still cannot train a preference model.
