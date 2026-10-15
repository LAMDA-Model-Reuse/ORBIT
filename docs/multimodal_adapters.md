# RouteFM-Qwen and ProfileRouter multimodal adapters

The dataset, `train`/`predict`/`evaluate` interfaces and metric definitions remain
unchanged. `Embedder.run_embed(texts=..., images=...)` now additionally supports
an explicitly selected joint encoder and aligned weighted fusion. Existing
methods still concatenate their independent feature vectors by default.

## RouteFM-Qwen

Text-only runs still default to the released RouteFM-BGE checkpoint. To use
the released Qwen router on a multimodal benchmark:

```bash
python main.py --dataset VLRouterBench --method RouteFM \
  --config configs/examples/routefm_qwen.json --device cuda:0
```

`--config` applies **top-level replacements** after benchmark/router defaults;
it does not recursively merge nested objects. `--device` takes final precedence.
Prepare the benchmark's upstream exports as described in
[VL-RouterBench](vl_routerbench.md) and configure its dataset directory first.
Set `routefm.checkpoint` to a local official Qwen router checkpoint, or leave it
null to use the pinned official Hub release. A BGE checkpoint is rejected.

The Qwen router requires **Qwen/Qwen3-VL-Embedding-8B, 4096 dimensions and L2
normalization**, not Qwen3 text embeddings, CLIP features, padded BGE vectors or
concatenated independent text/image features. The question and its image are
encoded together. Context and target use the same encoder and preprocessing;
only training observations populate Context. No held-out outcomes enter encoder
inputs. The frozen router and ORBIT's existing context cost transformation and
shared evaluation remain unchanged; predicted costs are not certified bills.

### Local encoder

The example uses `embeddings.qwen.backend="local"`. Download the full
Qwen3-VL-Embedding-8B encoder separately and set `embeddings.qwen.model_dir`.
This is an **8B encoder**, distinct from the much smaller RouteFM router weight.
Local precision defaults to the checkpoint dtype (`qwen.dtype="auto"`); override
with `float32`, `float16` or `bfloat16` when appropriate for your hardware.
The loader uses the [model-card SentenceTransformer interface](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B),
passing each input as `{"text": question, "image": local_image_path}`. Text-only
and image-only inputs are supported by the encoder too. Loading is local-only
with remote code disabled; there is no silent download or substitute encoder.
Use a compatible multimodal installation (the published SentenceTransformer
model metadata specifies Sentence Transformers 5.4.0; Transformers needs Qwen3-VL
support, e.g. 4.57.x under ORBIT's existing version range);
an old installation that cannot load the official model must be upgraded in a
separate encoder environment or replaced with the API backend below. The
existing main requirements are not globally upgraded for other methods.

### Explicit embedding service

Alternatively replace the `qwen` object in the example with:

```json
{
  "backend": "api",
  "base_url": "https://YOUR-SERVICE/v1",
  "model": "YOUR-PINNED-QWEN3-VL-EMBEDDING-8B-DEPLOYMENT",
  "api_key_env": "ROUTEFM_EMBEDDING_API_KEY",
  "timeout": 60
}
```

Set the named environment variable; never store API keys in a committed config.
This **explicitly uploads questions and image bytes** to the configured service
and may incur provider charges. API mode requires a RouteFM-compatible service
accepting `POST BASE_URL/embeddings` with this exact request contract:

```json
{
  "model": "DEPLOYMENT",
  "input": [{"text": "Question", "image": "data:image/png;base64,..."}]
}
```

The service must jointly encode each object and return indexed
`{"data":[{"index":0,"embedding":[...]}]}` rows. An optional `qwen.instruction`
is passed as `input[i].instruction` (local mode uses the model's `prompt`
argument); without it the encoder/provider must use its official default.
This image-input format is **not a universal OpenAI or vLLM REST schema**. Use a
compatible provider or a wrapper around the official encoder; a generic text
embedding endpoint is insufficient. Pin the served encoder revision and
preprocessing across all Context/target calls. The client cannot independently
certify what weights a remote provider runs.

Images must be existing local files; API uploads accept JPEG/PNG/WebP, detect
MIME from image contents (including `.image` files), and reject missing images
and uploads over 32 MiB per image. Remote image URLs are not fetched. HTTPS is
required except for a loopback service, redirects are rejected, and endpoint
credentials in the URL are forbidden. Row identities, dimensions, finite values
and nonzero norms are checked, and output is L2-normalized. Failures never
silently switch to text-only routing.

See the [official RouteFM custom-data contract](https://github.com/LAMDA-Model-Reuse/RouteFM/blob/main/docs/CUSTOM_DATA.md).

## ProfileRouter: aligned multimodal extension

```bash
python main.py --dataset VLRouterBench --method ProfileRouter \
  --config configs/examples/profile_multimodal.json --device cuda:0
```

The default text-only Longformer/profile path remains unchanged. Image mode
selects the matched pretrained CLIP ViT-B/16 text and image encoders. Public
model, family and task descriptions are encoded with its text branch, and the
existing training-free graph propagation produces 512-D model profiles. A query
uses, with each branch separately L2-normalized:

`q = normalize(alpha * text(question) + (1-alpha) * image(picture))`

The default `alpha=0.5` is fixed before evaluation. Configure it through
`profile.multimodal_embeddings.text_weight` with `0 < alpha < 1`; do not tune
against test outcomes. Query and profile remain in the same aligned space and
dimension. Cosine scoring and the shared, training-only cost MLP are unchanged.
Description-only calls do not require dummy images. Unpaired encoder names,
unequal dimensions, random trainable projections and invalid weights are
rejected. Custom encoder pairs under `profile.multimodal_embeddings` must
actually share a pretrained aligned space; equal vector sizes alone are not
proof of alignment. CLIP's text tokenizer truncates long descriptions/prompts
to its context window, unlike the original Longformer path.

This is an **ORBIT multimodal extension**, not an original-paper multimodal
reproduction. It adds no query-reward supervision to profile construction and
does not replace descriptions with held-out benchmark outcomes. Original
[RouteProfile](https://arxiv.org/abs/2605.00180) results and this extension should
be reported separately.

## Smoke boundaries

```bash
# Offline: all 28 image-compatible routers, tiny official Qwen architecture.
python scripts/smoke_all_methods.py --benchmarks VLRouterBench --multimodal \
  --methods Avengers AvengersPro CarrotRouter EARAMRouter Eagle EmbedLLM \
  EquiRouter GraphRouter HybridLLM InferenceDynamics MIRT NIRT OmniRouter \
  ProfileRouter RMClassification RMInterval RMSoftmax RouteFM RouteLLM_MF \
  RouteLLM_SWRanking RouterDC SaveRouter TRouter UniRoute knn mlp oracle svm \
  --tiny-routefm --report-dir /tmp/orbit-vl-multimodal-smoke

# Real bounded source records and released Qwen router weights, NOT real Qwen
# semantic embeddings: input-boundary surrogates remain explicitly recorded.
python scripts/smoke_all_methods.py --real-data --vl-source /path/to/prepared \
  --benchmarks VLRouterBench --multimodal --methods RouteFM ProfileRouter \
  --routefm-checkpoint /path/to/official/routefm_qwen.pt \
  --report-dir /tmp/orbit-vl-qwen-profile-smoke
```

The smoke runner uses a deterministic 4096-D text/image-dependent surrogate at
Qwen's encoder boundary, not genuine Qwen embeddings. Profile smoke uses
dimension-matched hash/pixel surrogates, not actual aligned CLIP semantics.
Router/context construction, cost prediction, allocation, metrics and JSON
writing are not mocked. Reports record substitutions. Unit tests separately
exercise real loopback HTTP transport, request image contents, row order and
input validation. A substituted run cannot certify production encoder loading
or routing quality. For real pretrained semantics, use `main.py` with the
examples above and the required encoder resources.

ModelSAT and RouteLLM_BERT remain outside the multimodal smoke list. CarrotRouter
and EA-RAM are included; EA-RAM's optional response-based ex-post evaluator is
not fitted on VL-RouterBench exports without released response strings.
