### Dataset description

The dataset was downloaded (upon reboot of the hosting server, thanks to some of the manuscript authors) from http://meltomeatlas.proteomics.wzw.tum.de:5003 .

It contains "melting behaviour" and a "melting degree" for proteins in human and other species (in two files).

### Full dataset

The full dataset (`full_dataset.json`) contains a list of proteins with the following structure:
```json
[
    {
      proteinId: [string from dataset],
      uniprotAccession: [if can be extracted],
      runName: [string from dataset],
      meltingPoint: [melting tempertature in C; may be null],
      quantNormMeltingPoint: [melting temp in C (for human sequences only); may be null]
      meltingBehaviour: [
        {
          tempertaure: [temperature in C],
          fold_change: [float usually between 0 and 1 (but can be higher or lower due to experimental weirdness)],
          channel: [string; present only for non-human]
        },
        {
          tempertaure: temp,
          fold_change: fchg,
          channel: channel
        },
        # ... [usually 10]
      ]
    },
    # ...
]
}
```

The combination of `proteinId` and `runName` may be used to uniquely identify a protein in one experiment, but unfortunately this doesn't always seem to hold true.

All sequences in the set where exported as fasta in the `full_dataset_sequences.fasta` file.
This file, in turn, was used to cluster sequences at 20% sequence identify via MMSeqs2 like so:

```
mmseqs easy-cluster --min-seq-id 0.2 meltome.fasta meltome_PIDE20.fasta tmp
```

The cluster file `meltome_PIDE20_clusters.tsv` was used to create train/test splits.

### Tasks

The train/test splits were computed by splitting the clusters. 80% of the clusters were used for training, while 20% were used for training. Additionally, either cluster representatives or cluster components were used for either set like so: 

- `clustered_task`: uses only the cluster representative from the mmseqs2 run for training and testing (aka: lower number of samples).
- `full_task`: uses cluster components and representatives for training and testing
- `mixed_task`: uses cluster components for training and cluster representatives for testing (goal: avoid overestimaiting performance on big clusters)


All tasks are contained in the `tasks.zip` file. These are CSV with colums:

- `sequence`: the AA sequence. May contain special characters!
- `target`: the prediction target. This may be continuous (`regression`), or True/False (`binary`)
- `set`: either `train` or `test`, if the sequence should be used for training or testing your model!


### Cite
From the publisher:
> Jarzab, A., Kurzawa, N., Hopf, T. et al. Meltome atlas—thermal proteome stability across the tree of life. Nat Methods 17, 495–503 (2020). https://doi.org/10.1038/s41592-020-0801-4


Bibtex:
```
@article{jarzab2020meltome,
  title={Meltome atlas—thermal proteome stability across the tree of life},
  author={Jarzab, Anna and Kurzawa, Nils and Hopf, Thomas and Moerch, Matthias and Zecha, Jana and Leijten, Niels and Bian, Yangyang and Musiol, Eva and Maschberger, Melanie and Stoehr, Gabriele and others},
  journal={Nature methods},
  volume={17},
  number={5},
  pages={495--503},
  year={2020},
  publisher={Nature Publishing Group}
}
```

### Data licensing

Upon request, the authors of the manuscript wish to make it known that the original dataset obtainable from http://meltomeatlas.proteomics.wzw.tum.de:5003 is:

> is free for anyone to use.
> No need for licenses
> Referencing/acknowledging the originators of the data would be good enough for me.

The modified data available in this repository and in the `tasks` falls under [AFL-3](https://opensource.org/licenses/AFL-3.0).