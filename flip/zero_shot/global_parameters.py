"""Global parameters used for zero-shot prediction"""

import re
import string

import numpy as np  # For setting global parameters, pylint: disable=unused-import
import torch


# Define a tuple of all amino acids
ALL_AAS = (
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
)
ALLOWED_AAS = frozenset(ALL_AAS)

# Set the default device to be used by PyTorch
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The default indexing system assumed to be used to name mutations
DEFAULT_INDEXING_SYSTEM = 1

# Turn tqdm on and off
DISABLE_TQDM: bool = False

# Define a regex pattern for finding mutation formatting
MUTATION_FINDER_RE = re.compile("([A-Z])([0-9]+)([A-Z])$")


def _build_msa_translation():

    # This is an efficient way to delete lowercase characters and insertion characters from a string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    return str.maketrans(deletekeys)


MSA_TRANSLATION = (
    _build_msa_translation()
)  # A translation object useful for processing MSAs
