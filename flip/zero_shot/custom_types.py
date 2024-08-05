"""Custom types for zero-shot predictions."""

from typing import Optional

import numpy as np
import numpy.typing as npt

# Array types
FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]

# Type for parsed mutations
ParsedMutant = Optional[list[tuple[str, int, str]]]

# Type for list of sequences
SequenceList = list[str]

# Basic MSA type
MsaType = list[tuple[str, str]]

# Old to new dictionary
OldToNew = dict[int, int]
