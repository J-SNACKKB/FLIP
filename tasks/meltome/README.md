Full dataset contains a list of proteins with the following structure:
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

All sequences in the set where exported as fasta in the `full_dataset_sequences.fasta` file.
This file, in turn, was used to cluster sequences at 20% sequence identify via MMSeqs2 like so:

```

```