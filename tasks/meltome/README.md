### Dataset description

The dataset was downloaded (upon reboot of the hosting server) from http://meltomeatlas.proteomics.wzw.tum.de:5003 .

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