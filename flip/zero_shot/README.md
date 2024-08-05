Use this module to make zero-shot predictions with ESM models. Examples follow:

```python
from flip.zero_shot import EsmPredictor, EsmMsaPredictor, MSA

# To make predictions using a sequence-based model
parent_sequence = "ACDEFGHIKLMNPQRSTVWY"
mutants = [
    "A1D", # Single mutant
    "C2F, Y20A", # Double mutant, etc.
    ...
]
model = EsmPredictor.esm2_t12_35M_UR50D(device = "cuda: 0")
predictions = model.predict(parent_sequence, mutants)

# To make predictions using an MSA-based model
parent_msa = MSA("path/to/msa.a3m") # .a2m also accepted
model = EsmMsaPredictor.esm_msa1b_t12_100M_UR50S(device = "cuda:0")
predictions = model.predict(parent_msa, mutants)
```

The following sequence-based ESM models are available via the `EsmPredictor` class, all following the notation used in the above example:

- esm1_t34_670M_UR50S
- esm1_t34_670M_UR50D
- esm1_t34_670M_UR100
- esm1_t12_85M_UR50S
- esm1_t6_43M_UR50S
- esm1b_t33_650M_UR50S
- esm2_t6_8M_UR50D
- esm2_t12_35M_UR50D
- esm2_t30_150M_UR50D
- esm2_t33_650M_UR50D
- esm2_t36_3B_UR50D
- esm2_t48_15B_UR50D
- esm1v_t33_650M_UR90S_1
- esm1v_t33_650M_UR90S_2
- esm1v_t33_650M_UR90S_3
- esm1v_t33_650M_UR90S_4
- esm1v_t33_650M_UR90S_5

The following MSA-based ESM models are available via the `EsmMsaPredictor` class, again following the notation used in the example above:

- esm_msa1_t12_100M_UR50S
- esm_msa1b_t12_100M_UR50S
