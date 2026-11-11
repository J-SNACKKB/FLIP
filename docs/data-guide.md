# Working with FLIP2 data

[Back to the repository overview](../README.md)

Use the [project website](https://flip.protein.properties/) for interactive dataset
descriptions and the [Zenodo record](https://doi.org/10.5281/zenodo.18433203) for the
archived release. FLIP2 files are hosted separately from this repository's legacy `splits/`.

## Download a split

Each link below downloads a gzip-compressed CSV. Filenames match the website release.
Read the [split definitions](https://flip.protein.properties/#splits) and dataset
provenance before interpreting results; different datasets have different target scales.

| Dataset | Split files |
| --- | --- |
| Amylase | [one_to_many](https://flip.protein.properties/assets/splits/amylase/one_to_many.csv.gz), [close_to_far](https://flip.protein.properties/assets/splits/amylase/close_to_far.csv.gz), [far_to_close](https://flip.protein.properties/assets/splits/amylase/far_to_close.csv.gz), [by_mutation](https://flip.protein.properties/assets/splits/amylase/by_mutation.csv.gz) |
| IRED | [two_to_many](https://flip.protein.properties/assets/splits/ired/two_to_many.csv.gz) |
| NucB | [two_to_many](https://flip.protein.properties/assets/splits/nucb/two_to_many.csv.gz) |
| TrpB | [one_to_many](https://flip.protein.properties/assets/splits/trpb/one_to_many.csv.gz), [two_to_many](https://flip.protein.properties/assets/splits/trpb/two_to_many.csv.gz), [by_position](https://flip.protein.properties/assets/splits/trpb/by_position.csv.gz) |
| Hydro | [three_to_many](https://flip.protein.properties/assets/splits/hydro/three_to_many.csv.gz), [low_to_high](https://flip.protein.properties/assets/splits/hydro/low_to_high.csv.gz), [to_P06241](https://flip.protein.properties/assets/splits/hydro/to_P06241.csv.gz), [to_P0A9X9](https://flip.protein.properties/assets/splits/hydro/to_P0A9X9.csv.gz), [to_P01053](https://flip.protein.properties/assets/splits/hydro/to_P01053.csv.gz) |
| Rhomax | [by_wild_type](https://flip.protein.properties/assets/splits/rhomax/by_wild_type.csv.gz) |
| PDZ3 | [single_to_double](https://flip.protein.properties/assets/splits/pdz3/single_to_double.csv.gz) |

## CSV schema and validation rows

| Column | Meaning |
| --- | --- |
| `sequence` | Amino acid sequence supplied to the model |
| `target` | Experimental target value; may be continuous or binned |
| `set` | Partition label, such as `train`, `test`, or `validation` |
| `validation` | Boolean flag identifying held-out validation rows |

**A row with `set=train` can still be a validation example.** Exclude rows marked
`validation=True` from training. The [quick-start example](../README.md#quick-start)
handles both the flag and an explicit `set=validation` label.

Fit preprocessing and model parameters on training data only. Use validation data for
model selection, and reserve test data for final evaluation. Do not replace a published
split with a random split when reporting benchmark comparisons.

For a local copy of a CSV, pandas can decompress it directly:

```python
import pandas as pd

df = pd.read_csv("one_to_many.csv.gz")
```

The website also documents [FASTA metadata](https://flip.protein.properties/#data-formats)
for sequence-based workflows. Check the actual header format of the chosen release.

## Reporting an experiment

Record the dataset and split filename, release URL or archive version, download date,
file checksum, random seeds, preprocessing, and model configuration. Report partition
sizes after filtering and the evaluation metric used for that task. For paper comparisons,
follow the paper's task-specific protocol and cite the source experimental dataset.

## Research code

The scripts preserve the research workflows used for data preparation and baselines.
They are useful starting points, but do not yet form a single portable installation:

- [`setup.py`](../setup.py) currently lists `helpers` packages, while the checked-in
  utility code lives under `flip/`. The root environment includes an editable install
  of that package. Use the standalone data-loading example above to begin exploring data.
- [`baselines/README.md`](../baselines/README.md) documents the original FLIP baseline
  workflow. Its model and split list should not be read as the FLIP2 experiment matrix.
- Scripts such as [`linear_models.py`](../baselines/linear_models.py) reference
  experiment-specific filesystem paths and additional dependencies such as
  `sequence_models`. Inspect and adapt these paths before running an experiment.
- [`collect_splits/`](../collect_splits/) contains preprocessing scripts and notebooks;
  raw source data and local paths may be required to regenerate splits.

Use the released files for benchmark evaluation, and the
[paper](https://doi.org/10.64898/2026.02.23.707496) and
[results page](https://flip.protein.properties/#performance) for the FLIP2 methods and results.
