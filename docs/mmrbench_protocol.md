# MMR-Bench V1 data integrity and evaluation protocols

The production `MMRBench` loader reads the released CSV and image archive.
It converts literal `True`/`False` and numeric quality labels into numeric
values in [0,1], and preserves original monetary costs, including valid zeros.
Malformed/out-of-range labels and negative/infinite costs fail explicitly.
Missing observations are never converted to incorrect answers or zero costs.

## Candidate coverage

The historical default candidate order contains ten models. In the audited V1
release, GPT-5-nano has no score/cost for MathVision's 3,040 queries. The shipped
benchmark configuration now explicitly selects
`dataset.candidate_policy="complete-models"`: retain only candidates with
finite quality and cost on every query, in their original order. For that
release this means nine models and all 10,370 queries across seven datasets,
including MathVision and OCRBench. No query is dropped to complete the pool.
Unknown extra CSV candidates are not automatically added to the declared pool.

Excluded model names and coverage counts are logged and available through
`loader.excluded_models` and `loader.model_coverage`. Candidate selection
depends only on observation availability, never performance magnitude.
`dataset.candidate_policy="error"` rejects incomplete candidates instead;
programmatic configurations without a policy also default to this strict mode.
The loader fails if fewer than two complete candidates remain.

The corrected nine-model pool is not the historical ten-model evaluation.
Report the candidate policy and actual returned model list with results.

## Response compatibility and images

Each actual released answer is exposed as `model_i_response` for the shared
router/EA-RAM contract. The historical `model_i_prediction` field is retained
as an identical alias. Missing response strings are not fabricated; EA-RAM's
existing response-availability checks still apply.

Images may be in `DATASET_DIR/TASK/ID.ext` (the archive's layout) or in
`DATASET_DIR/images/TASK/ID.ext`. SEEDBench2_Plus maps to SEEDBenchv2Plus.
Missing or ambiguous images fail with the query ID, rather than yielding a
missing path. Joint grouping additionally decodes every image and rejects
unreadable files; it never substitutes a zero image.

## Optional joint image/text grouping

The default `dataset.protocol="legacy"` retains the previous seeded row-random
**in-domain split**, with a warning that repeated inputs can cross train/test.
It does not bypass label/cost validation or restore incomplete observations.

For a common split that also protects text-view baselines, opt in explicitly:

```bash
python main.py --dataset MMRBench --method knn \
  --config configs/examples/mmrbench_joint_grouped.json --device cuda:0
```

Set the example's dataset directory to your local release. `--config` replaces
top-level objects, so the example includes the complete `dataset` settings.

`dataset.protocol="joint-grouped"` forms connected components: queries with
identical decoded RGB images **or** identical whitespace-normalized text must
remain together, including transitive links. Different image encodings with
the same decoded pixels are grouped. Original prompts are not normalized for
encoder input, and quality, costs and answers are never used for grouping.
The configured seed and train fraction drive `GroupShuffleSplit` on these
components. The query fraction is approximate and the split is not per-task
stratified; a whole task can fall in test. Report this as a distinct protocol,
not the untouched historical split or a guaranteed within-task evaluation.

This grouping changes only MMR-Bench V1 in-domain splitting. Out-of-domain task
splitting, other benchmarks, method algorithms and metric definitions are
unchanged. It prevents input overlap across the experiment split, not exposure
during external pretrained-model training.
