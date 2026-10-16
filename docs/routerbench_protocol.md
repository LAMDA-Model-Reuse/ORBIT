# Routerbench input-consistent evaluation

The standard Routerbench configuration now explicitly selects
`dataset.protocol="grouped-clean"`. This corrects two problems in the earlier
loader: taking the first prompt could silently combine candidate outcomes for
different inputs under the same query ID, and row-random splitting could put
identical query text in both train and test.

The clean protocol:

1. Checks original prompt equality across candidate records for each query ID.
   IDs with conflicting inputs are excluded before pivoting; their IDs are
   logged and exposed as `loader.input_conflict_sample_ids`. Exclusion depends
   only on input consistency, never quality, response, or cost values.
2. For in-domain splitting, groups queries by whitespace-normalized prompt and
   uses `GroupShuffleSplit` with the configured seed and train fraction. The
   original prompt strings are retained for encoding. The query fraction is
   approximate because whole groups must stay together.

Candidate order, performance values, original monetary costs, and metric
definitions are unchanged. This is a corrected data protocol, not the original
row-random protocol; identify it explicitly when reporting results. It prevents
cross-split input overlap, not exposure during an external model's pretraining.
Other benchmarks and Routerbench's out-of-domain task split are unchanged.

## Reproduce historical loading and splitting

Set `dataset.protocol="legacy"` in the Routerbench benchmark configuration to
retain first-prompt aggregation and the original seeded row-random in-domain
split. Warnings identify conflicting-input merges and the potentially leaky
split. Use this only when reproducing historical results, not as a clean
evaluation. Other dataset options must be retained when replacing the
`dataset` object through a top-level `--config` override.

## RouteFM context

The default RouteFM adapter uses **every training row** as observed Context,
with query rows aligned across all candidates. Test queries are Target only;
their observed outcomes are used only by evaluation. An integer
`routefm.context_size` requests an explicit fixed-K ablation using the same
training query rows for each candidate. Missing candidate observations are
masked, not filled with fictitious observations. Test-query batch size changes
memory usage, not the amount of training Context.
