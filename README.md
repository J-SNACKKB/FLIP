<h1 align="center">FLIP2</h1>

<p align="center"><strong>Protein fitness benchmarks for real-world generalization</strong></p>

<p align="center">
  <a href="https://flip.protein.properties/">Website</a> ·
  <a href="https://doi.org/10.64898/2026.02.23.707496">Paper</a> ·
  <a href="https://doi.org/10.5281/zenodo.18433203">Data archive</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="docs/data-guide.md">Dataset guide</a> ·
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <a href="https://flip.protein.properties/"><img src="https://img.shields.io/badge/FLIP2-7_datasets_%C2%B7_16_splits-167D8D" alt="FLIP2: 7 datasets, 16 splits"></a>
  <a href="https://doi.org/10.5281/zenodo.18433203"><img src="https://img.shields.io/badge/data-Zenodo-1682D4" alt="Data on Zenodo"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/code_license-AFL--3.0-blue" alt="Code license: AFL-3.0"></a>
</p>

How well do protein sequence models generalize beyond the variants they were trained on?
**FLIP2** expands the original **FLIP** benchmark with seven datasets and sixteen predefined
splits covering enzymes, protein interactions, stability, and light-sensitive proteins.

<p align="center">
  <a href="https://flip.protein.properties/#overview"><img src="https://flip.protein.properties/assets/paper_figures/fig1-1.png" width="1000" alt="FLIP2 overview: protein datasets, generalization splits, and baseline prediction methods."></a>
</p>

*Figure 1 from the [FLIP2 paper](https://doi.org/10.64898/2026.02.23.707496). Explore the
[interactive project page](https://flip.protein.properties/) for structures and results.*

## Start here

| I want to… | Start with… |
| --- | --- |
| Evaluate a model on FLIP2 | [Load a split](#quick-start), then use the [dataset guide](docs/data-guide.md) |
| Download the benchmark | [Individual split downloads](https://flip.protein.properties/#downloads) or the [Zenodo archive](https://doi.org/10.5281/zenodo.18433203) |
| Understand the experiments | [Paper](https://doi.org/10.64898/2026.02.23.707496) and [baseline results](https://flip.protein.properties/#performance) |
| Explore preprocessing or baseline code | [Repository map](#repository-map) and [research-code notes](docs/data-guide.md#research-code) |
| Use the original FLIP tasks | [Legacy FLIP](#legacy-flip) |

## Quick start

You only need Python and pandas to load the released CSV files; a GPU and a repository
installation are unnecessary for this example. In a fresh environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install pandas
```

Load one published split directly from the website:

```python
import pandas as pd

url = "https://flip.protein.properties/assets/splits/amylase/one_to_many.csv.gz"
df = pd.read_csv(url)

# Validation rows may also have set="train". Exclude them before fitting.
is_validation = df["validation"].astype(str).str.lower().eq("true")
is_validation |= df["set"].eq("validation")
train = df.loc[df["set"].eq("train") & ~is_validation].copy()
validation = df.loc[is_validation].copy()
test = df.loc[df["set"].eq("test") & ~is_validation].copy()

print({"train": len(train), "validation": len(validation), "test": len(test)})
# Use train["sequence"] and train["target"] to fit your model.
# Select hyperparameters on validation; evaluate on test only at the end.
```

For reproducible experiments, keep a local copy and record its source, download date,
and checksum. The [data guide](docs/data-guide.md) describes the columns, split filenames,
and how to report results. Preserve the released assignments when comparing models.

## FLIP2 datasets

| Dataset | Prediction task | Splits |
| --- | --- | ---: |
| Amylase | Starch stain removal activity | 4 |
| IRED | Imine reductase activity | 1 |
| NucB | Nuclease activity bins | 1 |
| TrpB | Tryptophan synthase fitness | 3 |
| Hydro | Hydrophobic-core stability | 5 |
| Rhomax | Rhodopsin absorption wavelength | 1 |
| PDZ3 | Protein interaction fitness | 1 |

The split families probe generalization across **mutation counts**, **sequence positions**,
**mutation identities**, **fitness levels**, and **wild-type backgrounds**. See the
[dataset guide](docs/data-guide.md#download-a-split) for all sixteen files and the
[website](https://flip.protein.properties/#splits) for the experimental rationale.

## Repository map

| Path | Contents |
| --- | --- |
| [`collect_splits/`](collect_splits/) | Dataset processing scripts and notebooks |
| [`baselines/`](baselines/) | Baseline training, embeddings, and zero-shot scoring code |
| [`analysis/`](analysis/) | Result analysis and plotting |
| [`flip/`](flip/) | Sequence utilities and zero-shot helpers |
| [`examples/`](examples/) | Existing example notebooks |
| [`splits/`](splits/) | Legacy FLIP archives, dataset descriptions, and split status |

**FLIP2 data are distributed through the website and Zenodo.** The checked-in `splits/`
directory contains legacy tasks. Research scripts may require local data paths, model
dependencies, and compute resources; see [research-code notes](docs/data-guide.md#research-code)
before running them.

## Legacy FLIP

The original FLIP benchmark remains available here. Start with the
[split index](splits/README.md), including [AAV](splits/aav/README.md),
[GB1](splits/gb1/README.md), and [Meltome](splits/meltome/README.md).
Historical [raw data](http://data.bioembeddings.com/public/FLIP) and
[FASTA downloads](http://data.bioembeddings.com/public/FLIP/fasta/) are hosted separately;
the FASTA representation follows [biotrainer](https://github.com/sacdallago/biotrainer).

### Split semaphore

The legacy dataset READMEs use the following status indicators:

- 🟢 **Active:** suitable for model evaluation.
- 🟠 **Not for performance comparisons:** may overestimate performance or duplicate the discriminative value of an active split.
- 🔴 **Obsolete:** do not use to report performance.

## Citation

Please cite [FLIP2](https://doi.org/10.64898/2026.02.23.707496) when using the new benchmark,
and the [original FLIP paper](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/8c19f571e251e61cb8dd3612f26d5ecf-Abstract-round2.html)
when using its tasks. Cite the underlying experimental datasets as well; provenance and
attribution are described on the [project website](https://flip.protein.properties/#datasets)
and in the data release.

```bibtex
@article{didi2026flip2,
  title = {FLIP2: Expanding Protein Fitness Landscape Benchmarks for Real-World Machine Learning Applications},
  author = {Didi, Kieran and Alamdari, Sarah and Lu, Alex X. and Wittmann, Bruce and Johnston, Kadina E. and Amini, Ava P. and Madani, Ali and Czeneszew, Maya and Dallago, Christian and Yang, Kevin K.},
  journal = {bioRxiv},
  year = {2026},
  doi = {10.64898/2026.02.23.707496}
}

@inproceedings{dallago2021flip,
  title = {FLIP: Benchmark tasks in fitness landscape inference for proteins},
  author = {Dallago, Christian and Mou, Jody and Johnston, Kadina E. and Wittmann, Bruce J. and Bhattacharya, Nicholas and Goldman, Samuel and Madani, Ali and Yang, Kevin K.},
  booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
  year = {2021}
}
```

GitHub's **Cite this repository** menu is supported by [`CITATION.cff`](CITATION.cff).

## Contributing and licensing

For dataset questions, reproducibility problems, or documentation fixes, open an
[issue](https://github.com/J-SNACKKB/FLIP/issues) or a pull request. See
[CONTRIBUTING.md](CONTRIBUTING.md) for useful details to include.

Repository code is covered by the existing [AFL-3.0 license](LICENSE).
The FLIP2 website describes its redistributed data release as **CC-BY 4.0**; consult the
release and individual dataset provenance for source-specific attribution and licensing.
