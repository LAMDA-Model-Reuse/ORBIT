# VL-RouterBench in ORBIT

ORBIT supports [VL-RouterBench](https://github.com/K1nght/VL-RouterBench) through
the existing `BaseDatasetLoader` contract, retaining the standard router,
training and evaluation interfaces. Dataset adaptation does not change metrics.

The upstream benchmark describes 14 datasets and 17 VLM candidates. Its public
[Hugging Face release](https://huggingface.co/datasets/KinghtH/VL-RouterBench/tree/6e44fbf50e170f8e72734b49f63f2b52a8866377)
is a roughly 6.87 GB raw archive, **not** a tabular Hugging Face routing dataset.
The HF viewer displays the repository's illustrations, not the routing records.
One-time upstream preparation is therefore required.

## Prepare the data

Install ORBIT's normal dependencies. The **upstream preparation** tools also
require `openpyxl`, `tiktoken` and `PyYAML` (see upstream requirements/setup).
They are not new ORBIT runtime dependencies: the loader reads JSONL, Parquet,
CSV and TSV exports and does not import the upstream package.

Use a new or empty preparation directory:

If a download was interrupted, a directory containing only the helper's `hub`
cache may be reused for resumption; existing extracted/prepared data is never
overwritten by the helper.

```bash
# Hash-check the frozen archive and extract with ORBIT's validated TAR helper.
# Reserve space for both the archive and the expanded image/inference assets.
python scripts/download_vlrouterbench.py --output-dir ./data/vl_routerbench

# Or reuse a previously downloaded official archive:
python scripts/download_vlrouterbench.py \
  --archive /path/to/vlm_router_data.tar.gz \
  --output-dir ./data/vl_routerbench

# Prepare official named export tables, using the reviewed upstream revision.
git clone https://github.com/K1nght/VL-RouterBench.git /path/to/VL-RouterBench
git -C /path/to/VL-RouterBench switch --detach f5a41ce83d6efe67489b35bb7a07ae72976a9aba
```

From the upstream checkout, with an **absolute** path to your preparation root:

```bash
# Step 2 resolves the model/pricing config relative to the preparation root.
cp -r config /path/to/ORBIT/data/vl_routerbench/config

python tools/build_benchmark_from_vlmevalkit.py \
  --vlmevalkit_dir /path/to/ORBIT/data/vl_routerbench/vlm_router_data/VLMEvalKit_evaluation \
  --tsv_base_dir /path/to/ORBIT/data/vl_routerbench/vlm_router_data/TSV_images \
  --config_dir config \
  --output_dir /path/to/ORBIT/data/vl_routerbench

python tools/compute_token_stats.py \
  --dataset_dir /path/to/ORBIT/data/vl_routerbench \
  --vlmevalkit_dir /path/to/ORBIT/data/vl_routerbench/vlm_router_data/VLMEvalKit_inference \
  --output_dir /path/to/ORBIT/data/vl_routerbench/reports/token_statistics
```

The upstream scripts are separate software: review their dependency and input
handling before running them. They may consume serialized raw evaluation files;
only use the verified official release or sources you trust. ORBIT's loader
and download helper **never unpickle data or import/execute upstream code**.
Steps 3–6 are not required for the ORBIT adapter; it never reads `Y.npz`, `C.npy`
or pickle index files, so positional matrix alignment and default-cost fallbacks
cannot silently enter the ORBIT evaluation.

Expected preparation root:

```text
BENCHMARKS/<task_type>/<dataset>_samples.jsonl
ORACLE/score/<dataset>.parquet
SPLITS/{train,dev,test}.jsonl
reports/token_statistics/token_stats_per_sample.csv
reports/token_statistics/output_tokens_per_model.csv
vlm_router_data/TSV_images/<dataset>.tsv
```

Missing exports produce an actionable error; the loader does not silently
download the complete archive, generate new VLM outcomes, or invoke paid APIs.
For an existing prepared directory, set `dataset.source_dir` in the benchmark
config. `dataset.dataset_dir` remains the output/cache/protocol directory.

## Run existing methods

```bash
python main.py --dataset VLRouterBench --method GraphRouter
python main.py --dataset VLRouterBench --method SaveRouter --device cpu
python main.py --dataset VLRouterBench --method oracle --device cpu
```

The default config is `configs/benchmarks/vlrouterbench.json`, with `text+image`
inputs and CLIP embeddings. Prepare the existing embedding checkpoints as usual.
Change `modality` to `text` and disable the image encoder only for an explicitly
labelled **text-only ablation**, including methods with text-only adapters.
Do not describe such runs as multimodal routing.

Optional settings under `dataset`:

- `models`: canonical IDs from `configs/datasets/vlrouterbench.json`.
- `benchmarks`: distinct source dataset IDs; all 14 are the default.
- `max_samples`: deterministic cap on input groups **per official train/test
  split**, applied before image materialization. It is for smoke/debugging,
  not a full-benchmark result.
- `source_dir`: preparation root; defaults to `dataset_dir`.
- `image_root`: directory containing the TSV image files; defaults to
  `<source_dir>/vlm_router_data/TSV_images`.
- `multi_image=contact-sheet`, `image_cell_size=512`: explicit multi-image
  transformation for ORBIT's existing single-image embedding input. This is not
  a claim to reproduce an upstream multi-image encoder's representation.
- `missing_policy=error` (default) or `drop-query`: reject incomplete candidate
  support or explicitly retain common complete support, recording exclusions.
- `text_overlap_policy=error` (default) or `drop-query`: text-only official views
  reject identical normalized prompts across train/dev/test by default.
  `drop-query` explicitly excludes all queries in those cross-split text groups,
  recording counts by split, **for both text and multimodal runs** so the same
  population filter can be shared when comparing methods.
  Different images do not make the text-only inputs distinguishable. The smoke
  runner explicitly selects `drop-query`; normal benchmark runs do not silently
  change the population. Alternatively use grouped ORBIT in-domain splits.

For fair comparisons, use the same candidate pool, population policy and actual
train/test ID sets for all methods. Bounded sampling operates on visible-input
groups, which differ between text-only and image views; check sidecar ID hashes
before comparing their scores, or use the complete shared official population.
- `estimated_output_policy=allow` (default), `error`, or `drop-query`: retain
  upstream output-length estimates with flags, require counted output lengths,
  or mark estimates missing and use `missing_policy=drop-query`. None of these
  options makes the input/image token estimates actual provider usage.

## Evaluation protocol and limitations

- All joins use source sample/model IDs. Model order is sorted and shared by
  every performance, cost, usage and metadata column. Duplicate identities,
  invalid scores, nonfinite/negative/fractional token counts and inconsistent
  input token totals fail explicitly. Missing scores/costs are never zero-filled.
- Performance is the released `quality_raw` value in `[0,1]`, without another
  integer cast or threshold. **All-zero-performance queries are retained**;
  `trainable_sample_ids.txt` is not consulted. ORBIT cannot undo any score
  conversion already performed by the upstream exporter.
- Official mode uses upstream `SPLITS/train.jsonl` and `test.jsonl`, excluding
  dev from supervision/evaluation. Duplicate assignments, unassigned samples
  and identical visible inputs across train/dev/test (declared prompt+image
  references for multimodal, normalized prompt alone for text-only)
  fail before bounded subsampling. The actual exported split IDs are hashed;
  this is not a claim that rerunning the upstream exporter regenerates the
  paper's original split (its shuffle also depends on source traversal order).
- Existing grouped `in-domain` and task-based `out-of-domain` ORBIT splits can
  be selected explicitly. Their results must be labelled as ORBIT protocols.
- Cost is `(total_input_tokens * input_price + output_tokens * output_price) /
  1e6`, using the frozen upstream reference prices, **not current API prices or
  verified provider bills**. Input/image tokenization is an estimate; output
  lengths may also be estimated. `model_i_output_is_actual`, token columns,
  `model_i_raw_cost`, source-table hashes and estimate counts remain available.
  `is_actual=True` only means upstream counted a released response instead of
  using a fallback length; it does **not** certify provider usage or per-model
  tokenization. Selected materialized image contents are also hashed in the sidecar.
  A single training-only maximum scales all cost columns by default.
- No gold answers, score-derived labels, model responses or cost information
  become router inputs. The standard exports do not retain response strings;
  EA-RAM's optional ex-post evaluator therefore cannot be trained from them.
  Its allocation/cost evaluation can still run using its normal no-response path.
- TSV image row references are **zero-based row ordinals** from the exporter,
  not numeric values of the TSV `index` field. Short image values reference
  another TSV index; these references are resolved separately, with missing,
  ambiguous and cyclic references rejected. Base64 image lists preserve order.
  Materialized images preserve the original encoded bytes; JPEGs are not
  inflated into PNG caches. PIL detects their format by content.
- Author-machine absolute TSV paths are relocated by basename under
  `image_root`. Direct image URIs must be valid relative paths under that root;
  absolute direct image paths require an explicit export repair. Remote image
  URLs and unsupported asset types do not silently degrade to blank images.
- ORBIT reports its existing nAUC, Peak Score and QNC. Upstream Rank Score and
  throughput are not replaced by these and are not claimed as reproduced.

## Local smoke

```bash
# All 30 methods, real training/prediction/metrics/JSON, source-schema fixtures.
python scripts/smoke_all_methods.py --benchmarks VLRouterBench --tiny-routefm \
  --report-dir /tmp/orbit-vl-text-smoke

# Image-compatible methods only; choose them through --methods.
python scripts/smoke_all_methods.py --benchmarks VLRouterBench --multimodal \
  --methods GraphRouter SaveRouter knn oracle \
  --report-dir /tmp/orbit-vl-image-smoke

# Prepared actual records, bounded to 32 train/test input groups and two models.
python scripts/smoke_all_methods.py --real-data --vl-source /path/to/prepared \
  --benchmarks VLRouterBench --methods GraphRouter SaveRouter knn oracle \
  --multimodal --report-dir /tmp/orbit-vl-real-smoke
```

The smoke runner uses deterministic input-only hash text features and actual
image-derived features. Large transformer checkpoints are replaced by tiny local
models and NIRT's paid annotation boundary is stubbed. These are runtime checks,
not paper-quality or full-benchmark reproduction claims. CI additionally tests
every registered method on all four source-schema fixtures, and 28 compatible
methods on each MMR V2/VL-RouterBench multimodal fixture.
RouteFM multimodal smoke uses Qwen router weights with an explicitly labeled
joint encoder surrogate; ProfileRouter uses aligned-fusion surrogates. For
production encoder configuration and limitations, see
[multimodal adapters](multimodal_adapters.md).
