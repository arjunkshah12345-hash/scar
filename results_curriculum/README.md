# Archived curriculum condition

These 27 JSON artifacts are the earlier parity-sparse runs trained with the
4→8→16→32 length curriculum. They are retained for transparency but are not
part of the five-condition release in `results/`, whose `parity_sparse`
condition is fixed at 32 operations with `curriculum: false`.

The release runner gives the optional curriculum condition a distinct
`*_parity_sparse_curriculum_seed*.json` identity so it cannot overwrite or be
mistaken for the fixed-length condition.
