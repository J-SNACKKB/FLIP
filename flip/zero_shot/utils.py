"""Utilities for zero-shot prediction"""

# pylint: disable=too-many-lines
import os.path

from copy import deepcopy
from typing import Optional, Sequence, Union

import numpy as np

from Bio import SeqIO

from .custom_types import MsaType, OldToNew, ParsedMutant
from .global_parameters import (
    ALLOWED_AAS,
    DEFAULT_INDEXING_SYSTEM,
    MSA_TRANSLATION,
    MUTATION_FINDER_RE,
)


def mutation_parser(
    mutstring: str, indexing_system: int = DEFAULT_INDEXING_SYSTEM
) -> ParsedMutant:
    """Checks a single mutation string to be sure that it complies with the
    format {ParentAA}{Ind}{MutantAA}. Returns that string split into a tuple
    formatted as (ParentAA, Ind, MutantAA). Note that indexing is adjusted to
    Python indexing (0-indexing) based on the value of `indexing_system` provided.

    Args:
        mutstring (str): The mutation string to be analyzed. This must take the
            form of space- and comma-separated {ParentAA}{Ind}{MutantAA} denoted
            mutations. For instance "V39G, D40G" would be a valid `mutstring`.
        indexing_system (int, optional): The indexing system used to define the
            amino acid index. This is used to convert protein indices into Python
            (0-indexed) indices. Defaults to DEFAULT_INDEXING_SYSTEM.

    Raises:
        ValueError: "Unexpected mutation formats identified" is raised if a mutation
            with a format not matching {ParentAA}{Ind}{MutantAA} is found.
        ValueError: "Parent and mutant amino acids cannot be the same in a mutation"
            is raised if the parent and mutant amino acids are the same.
        ValueError: "Duplicate mutation index found" is raised if two or more
            mutations share the same mutation index.

    Returns:
        tuple: A list of 3-member tuples consisting of the parent amino acid,
            the mutation index (now 0-indexed relative to a parent sequence),
            and the mutant amino acid for each mutation found in `mutstring`.
    """
    # If a parent sequence, we return an empty tuple.
    if mutstring == "":
        return None

    # Remove all whitespace
    mutstring = "".join(mutstring.split())

    # Capitalize
    mutstring = mutstring.upper()

    # Separate on commas
    all_muts = mutstring.split(",")

    # Make sure all mutations match the expected format
    mut_checks = [MUTATION_FINDER_RE.match(mut) for mut in all_muts]
    bad_mut_checks = [
        mut for mut, mut_check in zip(all_muts, mut_checks) if not mut_check
    ]
    if len(bad_mut_checks) != 0:
        raise ValueError(f"Unexpected mutation formats identified: {*bad_mut_checks,}")

    # Split apart the mutation
    mut_info = []
    mut_inds_found = []
    for mut_check in mut_checks:

        # Extract the different components of the mutation
        # assert mut_check is not None
        assert mut_check is not None
        parent_aa, mut_ind, mutant_aa = mut_check.groups()

        # The parent and mutant amino acids cannot be the same
        if parent_aa == mutant_aa:
            raise ValueError(
                "Parent and mutant amino acids cannot be the same in a mutation"
            )

        # Record as a tuple. We assume adjust the index based on the indexing system.
        if mut_ind in mut_inds_found:
            raise ValueError(f"Duplicate mutation index found: {mut_ind}")
        mut_inds_found.append(int(mut_ind))
        mut_info.append((parent_aa, int(mut_ind) - indexing_system, mutant_aa))

    # Mutation indices must be sorted
    ind_array = np.array(mut_inds_found)
    if not np.array_equal(ind_array, np.sort(ind_array)):
        raise ValueError("Mutation indices should be sorted!")

    return mut_info


def positions_from_multimutants(
    parent: str,
    mutations: Sequence[str],
    indexing_system: int = DEFAULT_INDEXING_SYSTEM,
) -> tuple[dict[int, str], list[ParsedMutant]]:
    """Identifies the unique mutated positions from a list of mutations.

    Args:
        parent (str): The parent sequence against which all mutations will be
            compared.
        mutations (Sequence[str]): A sequence of mutations to be analyzed. Each
            entry must take the form of space- and comma-separated
            {ParentAA}{Ind}{MutantAA} denoted mutations. For instance "V39G, D40G"
            would be a valid entry in the sequence.
        indexing_system (int, optional): The indexing system used to define
                the amino acid index. This is used to convert protein indices
                provided by `mutated_positions` into Python (0-indexed) indices.
                For instance, if `indexing_system = 1`, then the Python indices
                corresponding to D123 and L412 are 122 and 411. Defaults to
                DEFAULT_INDEXING_SYSTEM.

    Raises:
        ValueError: Raised if there is a mismatch between what is actually found
            in `parent` and the expected amino acid at the position given by an
            entry in `mutations`.

    Returns:
        tuple[dict[int, str], list[ParsedMutant]]: The first element is a dictionary
            mapping between  the position (0-indexed relative to the parent
            sequence) of the parent sequence and the expected amino acid at that
            position. The second element is the output of `mutation_parser` for
            each entry in `mutations`.
    """

    # Parse mutations
    mut_info = [
        mutation_parser(mutstring, indexing_system=indexing_system)
        for mutstring in mutations
    ]

    # Get unique positions and mutations
    pos_to_parent = {}

    for infoset in mut_info:
        for parent_aa, pos_ind, _ in infoset:

            # If we have seen this index before, make sure it matches
            # what we have already seen
            if pos_ind in pos_to_parent:
                assert pos_to_parent[pos_ind] == parent_aa

            # If we have not seen it, run some checks
            else:
                if parent[pos_ind] != parent_aa:
                    raise ValueError(
                        "Mismatch between parent in mutation and "
                        f"parent sequence: {parent[pos_ind]} != {parent_aa}"
                    )
                pos_to_parent[pos_ind] = parent_aa

    return pos_to_parent, mut_info


class MSA:
    """Loads and processes an MSA from an .a2m or .a3m file."""

    # Initialization is loading the msa from an a2m or a3m file
    def __init__(
        self: "MSA", msaloc: Optional[str] = None, _raw_msa: Optional[MsaType] = None
    ):
        """Loads a .a2m or .a3m file from disk and processes it. Processing means
        removing all "." and lowercase characters as well as checking for forbidden
        characters in the reference sequence.

        Args:
            msaloc (str): The path to the .a2m or .a3m file. Default = None.
        Raises:
            IOError: Raised if `msaloc` does not exist.
        """
        # We load the raw msa if it is not provided
        if msaloc is not None:

            # Make sure the location exists
            if not os.path.isfile(msaloc):
                raise FileNotFoundError(f"Could not find {msaloc} on disk")

            # Get the raw msa. This is adapted from code provided by ESM
            self._raw_msa = [
                (record.description, str(record.seq))
                for record in SeqIO.parse(msaloc, "fasta")
            ]

        # Otherwise we set the raw msa
        elif _raw_msa is not None:
            self._raw_msa = _raw_msa

        # Otherwise we have an error
        else:
            raise ValueError("Must provide `msaloc`")

        # Now we process the msa reference sequence to remove lowercase letters
        # and dots. We will also produce a dictionary that maps from old sequence
        # locations to new sequence locations in the processed MSA.
        (
            self._processed_query,
            self._original_query,
            self._old_to_new,
        ) = self._process_msa_refseq()

        # Get the indices that were not captured by the MSA
        self._missing_inds = tuple(
            ind
            for ind in range(len(self._original_query))
            if ind not in self.old_to_new
        )

        # Now we can process the full MSA like we did the reference sequence. We will
        # also make some sanity checks
        self._processed_msa = self._process_raw_msa()

        assert len(self.processed_msa) == len(self.raw_msa)

    def _process_msa_refseq(self: "MSA") -> tuple[str, str, OldToNew]:
        """Processes the query sequence for the MSA to both (1) reconstruct what
        the query sequence looked like before alignment (i.e. by capitalizing all
        lowercase characters and removing "." and "-" characters) and (2) produce
        a processed query sequence by removing all lowercase and "." characters.

        Raises:
            ValueError: Raised if a character other than an amino acid single letter
                code, ".", or "-" is encountered in the query sequence.

        Returns:
            tuple[str, str, OldToNew]: The first element is the query sequence with
                all "." and lowercase characters removed. The second element is
                the original query sequence prior to alignment (made by capitalizing
                lowercase characters and removing "." and "-" characters). The
                final element is a dictionary that maps indices in the second
                element to indices in the first element. Note that any positions
                deleted from the first element relative to the second do not show
                up in the dictionary.
        """

        # Now we loop over the reference sequence. Alphabetic characters increment
        # the original sequence counter; capital alphabetic characters and dashes
        # increment the new sequence counter. Dots have no effect on either counter.
        og_seq_counter: int = -1
        new_seq_counter: int = -1
        old_to_new: OldToNew = {}
        processed_refseq: list[str] = []
        original_refseq: list[str] = []
        for char in self._raw_msa[0][1]:

            # Check if this is an amino acid
            if char.upper() in ALLOWED_AAS:

                # Increment the original sequence counter
                og_seq_counter += 1

                # If it is an uppercase character, increment the new sequence
                # counter, add to the dictionary of stored characters, and update
                # the processed sequence
                if char.isupper():
                    new_seq_counter += 1
                    assert og_seq_counter not in old_to_new
                    assert new_seq_counter not in old_to_new.values()
                    old_to_new[og_seq_counter] = new_seq_counter
                    processed_refseq.append(char)

                # Whatever the character, add it to the original refseq
                original_refseq.append(char.upper())

            # If this is a dash, just increment the new seq counter and add
            # to the processed reference sequence
            elif char == "-":
                new_seq_counter += 1
                processed_refseq.append(char)

            # Otherwise this must be a ".". No other characters are allowed in the
            # reference sequence
            elif char != ".":
                raise ValueError(
                    f"Unexpected character in the MSA reference sequence: {char}"
                    " Allowed characters are single-letter amino acid codes, '.',"
                    " and '-'."
                )

        # The processed refseq should match the raw refseq at all entries in old-to-new
        assert all(
            original_refseq[old_ind] == processed_refseq[new_ind]
            for old_ind, new_ind in old_to_new.items()
        )

        return "".join(processed_refseq), "".join(original_refseq), old_to_new

    def _process_raw_msa(self: "MSA") -> MsaType:
        """Removes all "." and lowercase characters from every MSA element.

        Returns:
            MsaType: A list of 2-member tuples. The first member of each tuple
                gives the identity of the sequence while the second member gives
                the processed sequence.
        """
        # Process the msa
        processed_msa = [
            (name, MSA._remove_insertions(seq)) for name, seq in self._raw_msa
        ]

        # The query sequence should match what we saw when processing just the refseq
        assert self._processed_query == processed_msa[0][1]

        # Every sequence must be the same length
        query_seqlen = len(processed_msa[0][1])
        assert all(len(seq) == query_seqlen for _, seq in processed_msa)

        return processed_msa

    def remove_duplicates(self: "MSA") -> None:
        """Removes elements with duplicate sequences from self.processed_msa."""

        # Just loop over the processed msa and remove duplicates
        deduplicated_msa_inds = []
        observed_sequences = set()
        for seqind, (_, seq) in enumerate(self._processed_msa):
            if seq not in observed_sequences:
                deduplicated_msa_inds.append(seqind)
                observed_sequences.add(seq)

        # Assign the deduplicated MSA
        self._raw_msa = [self._raw_msa[ind] for ind in deduplicated_msa_inds]
        self._processed_msa = [
            self._processed_msa[ind] for ind in deduplicated_msa_inds
        ]

        # Make some checks
        assert len(self._raw_msa) == len(self._processed_msa)
        assert self.processed_query == self._processed_msa[0][1]

    def subsample_msa(
        self: "MSA", n_sampled: Union[np.integer, int], seed: int = 0
    ) -> "MSA":
        """Subsamples from the existing MSA without replacement, returning a new
        MSA object. The query sequence is always maintained as the first element.

        Args:
            n_sampled (int): The number of samples to take from the MSA. If this
                number is greater than or equal to the number of samples in the
                MSA, then a copy of the original MSA is returned.
            seed (int, optional): Seed for reproducibility. Defaults to 0.

        Returns:
            MSA: A new MSA object with as many samples in the raw and processed
                MSA as given by `n_sampled`.
        """

        # If we have less sequences than the number sampled requested, just return
        # a copy of the original object
        if len(self) < n_sampled:
            new_msa = deepcopy(self)

        # Otherwise, we sample randomly
        else:

            # Build our number generator
            rng = np.random.default_rng(seed)

            # To subsample, we choose a set of indices below the query sequence
            indices = np.sort(
                rng.choice(np.arange(1, len(self)), size=n_sampled - 1, replace=False)
            )

            # Grab the elements of the raw msa we want
            new_raw_msa = [self._raw_msa[0]] + [self._raw_msa[ind] for ind in indices]

            # Build the new MSA
            new_msa = MSA(_raw_msa=new_raw_msa)

            # Run checks to be sure our new MSA matches the old
            assert new_msa.original_query == self.original_query
            assert new_msa.processed_query == self.processed_query
            assert new_msa.old_to_new == self.old_to_new
            # pylint: disable=protected-access
            assert new_msa._missing_inds == self._missing_inds
            # pylint: enable=protected-access

        return new_msa

    @staticmethod
    def _remove_insertions(sequence: str) -> str:
        """Removes any insertions into the sequence. Needed to load aligned
        sequences in an MSA. This is taken directly from examples provided on the
        ESM GitHub repository."""
        return sequence.translate(MSA_TRANSLATION)

    @property
    def raw_msa(self: "MSA") -> MsaType:
        """The msa before it is processed to remove all lowercase and "." characters
        from each entry.
        """
        return self._raw_msa

    @property
    def original_query(self: "MSA") -> str:
        """The original query sequence before it was processed in an a3m or a2m
        file. This will consist entirely of uppercase amino acid characters.
        """
        return self._original_query

    @property
    def processed_query(self: "MSA") -> str:
        """The query sequence after processing from an a2m or a3m file to remove
        all "." and lowercase characters. This will consist only of uppercase
        amino acid characters and the "-" character.
        """
        return self._processed_query

    @property
    def old_to_new(self: "MSA") -> OldToNew:
        """A dictionary that maps indices in self.original_query to
        self.processed_query. Note that any positions deleted from self.original_query
        relative to self.processed_query do not show up in the dictionary.
        """
        return self._old_to_new

    @property
    def processed_msa(self: "MSA") -> MsaType:
        """The msa after it is processed to remove all lowercase and "." characters
        from each entry.
        """
        return self._processed_msa
