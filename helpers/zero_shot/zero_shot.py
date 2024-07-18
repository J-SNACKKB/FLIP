"""
Contains code that simplifies the interface to Meta's ESM for zero-shot prediction.
"""

from functools import wraps
from typing import Optional, Sequence, Union, overload, Callable, Type

import esm
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

from .custom_types import FloatArray, IntArray, ParsedMutant, SequenceList, MsaType
from .global_parameters import DEFAULT_DEVICE, DEFAULT_INDEXING_SYSTEM, DISABLE_TQDM
from .utils import MSA, positions_from_multimutants


def _load_esm_model(func: Callable):
    """
    A wrapper for instantiating children of `AbstractEsmPredictor` with various
    ESM models having either pretrained or random weights
    """

    @wraps(func)
    def inner(
        cls: Type["AbstractEsmPredictor"],
        pretrained: bool = True,
        device: str = DEFAULT_DEVICE,
    ):
        # pylint: disable=protected-access

        # Confirm the model is allowed
        assert isinstance(cls._allowed_models, dict)
        try:
            if func.__name__ not in cls._allowed_models:
                raise ValueError(
                    f"Unrecognized `model_name`: {func.__name__}. Must be one of "
                    f"{*cls._allowed_models,}."
                )
        except AttributeError as error:
            raise AttributeError(
                "Must define the class variable `_allowed_models`"
            ) from error

        # Get the model and its alphabet
        model, alphabet = cls._allowed_models[func.__name__]()

        # If we are not using a pretrained model, then replace the model with a
        # randomly instantiated one
        if not pretrained:
            return func(cls)

        else:
            return cls(
                model=model,
                alphabet=alphabet,
                device=device,
            )

    # Update the function signature
    inner.__doc__ = f"""
    Loads {func.__name__} with either pretrained or random weights.

    Args:
        pretrained (bool, optional): Whether or not to load a model with pretrained
            weights. Defaults to True.
        device (str, optional): The device on which the model will be run. Defaults
            to "cpu" if no GPUs are available, otherwise the first GPU is used.

    Raises:
        ValueError: Raised if an unknown model is requested.
        AttributeError: Raised if there is no class attribute `_allowed_models`
    """
    return inner


class AbstractEsmPredictor:
    """Abstract class that defines methods shared by all ESM models that will be
    used for making zero-shot predictions.
    """

    _is_msa_transformer = False
    _n_expected_tokenization_dims: Optional[int] = None
    _allowed_models: Optional[dict[str, Callable]] = None

    def __init__(
        self,
        model: Union[esm.MSATransformer, esm.ProteinBertModel, esm.ESM2],
        alphabet: esm.data.Alphabet,
        device: str,
    ) -> None:
        """
        Sets the model that will be used. This should not be called directly. Use
        the class methods to load the ESM model of choice.

        Parameters
        ----------
        model (Union[esm.MSATransformer, esm.ProteinBertModel, esm.ESM2]): The ESM
            model to use.
        alphabet (esm.data.Alphabet): The alphabet associated with the ESM model.
        device (str): The device on which the model will be run.
        """
        # The class variables must have been overwritten
        if self._n_expected_tokenization_dims is None:
            raise ValueError(
                "Must overwrite the class attribute `_n_expected_tokenization_dims` in child class"
            )
        if self._allowed_models is None:
            raise ValueError(
                "Must overwrite the class attribute `_allowed_models` in child classes"
            )

        # Assign instance variables
        self.model = model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.device = device

        # Note use
        self._uses_cls: bool = True
        self._uses_eos: bool = False

        # We move the model to eval mode
        self.model.eval()

    def predict(
        self,
        parent: Union[str, MSA],
        mutations: Sequence[str],
        indexing_system: int = DEFAULT_INDEXING_SYSTEM,
    ):
        """Makes zero-shot predictions using an ESM model by calculating the sum
        of log-odds ratios between mutant and parent amino acids.

        Args:
            parent (Union[str, MSA]): Either a string giving the sequence of a
                single parent protein sequence or an `MSA` instance that holds the
                MSA of the parent protein sequence. The string will be used for
                ESM models that work with protein sequences and the MSA will be
                used for ESM models that work with MSAs.
            mutations (Optional[Sequence[str]], optional): A sequence of mutations
                strings. Each mutation string must take the form of space- and
                comma-separated {ParentAA}{Ind}{MutantAA} denoted mutations. For
                instance, "V39G, D40G" would be a valid mutation string.
            indexing_system (int, optional): The indexing system used to define
                the amino acid index. This is used to convert protein indices
                provided by `mutated_positions` into Python (0-indexed) indices.
                For instance, if `indexing_system = 1`, then the Python indices
                corresponding to D123 and L412 are 122 and 411. Defaults to
                DEFAULT_INDEXING_SYSTEM.
        """
        # Check mutation signatures
        parent_aa_source = parent.original_query if self._is_msa_transformer else parent
        pos_to_parent, mut_info = positions_from_multimutants(
            parent=parent_aa_source,
            mutations=mutations,
            indexing_system=indexing_system,
        )

        # Build the log probability matrices
        pos_to_logprobs = self._get_mm_logprobs(
            parent=parent, pos_to_parent=pos_to_parent
        )

        # Calculate log odds ratios
        return self._get_mm_logodds(mut_info=mut_info, pos_to_logprobs=pos_to_logprobs)

    def _get_mm_logprobs(
        self,
        parent: Union[str, MSA],
        pos_to_parent: dict[int, str],
    ) -> dict[int, tuple[FloatArray, np.float32]]:

        # Build log probs
        pos_to_logprobs = {}
        n_positions = len(pos_to_parent)
        for seq_pos, parent_aa in tqdm(
            pos_to_parent.items(),
            total=n_positions,
            disable=DISABLE_TQDM,
            desc=f"Calculating log-probabilities for {n_positions} positions",
        ):

            # Update the sequence position
            target_positions = self._update_target_positions(
                np.array([seq_pos], dtype=int), parent=parent
            )
            assert target_positions.shape == (1,)

            # Get the log probabilities for the target position
            target_log_probs = self._get_masked_log_probs(
                parent=parent,
                target_positions=target_positions,
                parent_aas=(parent_aa,),
            )

            # Get the parent log prob
            assert target_log_probs.shape == (1, len(self.alphabet.tok_to_idx))
            parent_prob = target_log_probs[0, self.alphabet.tok_to_idx[parent_aa]]

            # Record information
            pos_to_logprobs[seq_pos] = (target_log_probs, parent_prob)

        return pos_to_logprobs

    def _get_mm_logodds(
        self,
        mut_info: list[ParsedMutant],
        pos_to_logprobs: dict[int, tuple[FloatArray, np.float32]],
    ) -> FloatArray:

        # Loop over all positions and calculate log odds ratios
        log_odds = np.zeros(len(mut_info), dtype=np.float32)
        for mutind, infoset in enumerate(mut_info):

            # Loop over all mutations in the set
            for _, pos_ind, mutant_aa in infoset:

                # Get the parent log probs and the log probs matrix
                log_probs, parent_log_prob = pos_to_logprobs[pos_ind]
                assert log_probs.shape == (1, len(self.alphabet.tok_to_idx))

                # Increment log odds
                log_odds[mutind] += (
                    log_probs[0, self.alphabet.tok_to_idx[mutant_aa]] - parent_log_prob
                )

        return log_odds[:, None]

    @overload
    def _get_masked_log_probs(
        self,
        parent: MsaType,
        target_positions: torch.Tensor,
        parent_aas: tuple[str, ...],
    ) -> FloatArray: ...

    @overload
    def _get_masked_log_probs(
        self,
        parent: str,
        target_positions: torch.Tensor,
        parent_aas: tuple[str, ...],
    ) -> FloatArray: ...

    def _get_masked_log_probs(self, parent, target_positions, parent_aas):
        """Gathers masked log probabilities for a parent sequence at a set of positions
        in combination.

        Args:
            parent (Union[MsaType, str]): The parent sequence or MSA for which
                masked log probabilities will be collected.
            target_positions (torch.Tensor): The positions that will be masked
                when gathering log probabilities.
            parent_aas (tuple[str, ...]): The expected identities of the amino
                acids at `target_positions` in either the processed query sequence
                of an MSA or the parent sequence, depending on whether a standard
                or MSA transformer is being used.

        Raises:
            ValueError: Raised if `target_positions` are not sorted.

        Returns:
            FloatArray: A numpy array of the masked log probabilities at the positions
                given by `target_positions` when those positions are masked. The
                shape of this array is the number of target positions by the number
                of characters in `self.alphabet`.

        """

        # Define variable types that need it
        inds_to_update_list: list[Union[int, torch.Tensor]]

        # Pull the processed MSA if we need to
        to_tokenize = parent.processed_msa if isinstance(parent, MSA) else parent

        # The target positions must be sorted
        assert len(target_positions.shape) == 1
        sorted_targets, _ = torch.sort(target_positions)  # pylint: disable=no-member
        if not torch.equal(  # pylint: disable=no-member
            sorted_targets, target_positions
        ):
            raise ValueError("`target_positions` must be sorted in ascending order")

        # Build the base tokenization
        base_tokenization = self._tokenize([to_tokenize])
        og_tokenization = base_tokenization.clone()
        assert isinstance(
            og_tokenization, torch.LongTensor  # pylint: disable=no-member
        )

        # Get the indices of the leading dimensions (i.e. all but the last)
        assert isinstance(self._n_expected_tokenization_dims, int)
        inds_to_update_list = [0] * (self._n_expected_tokenization_dims - 1)
        leading_dim_inds = tuple(inds_to_update_list)

        # Check the base tokenization
        assert len(base_tokenization.shape) == self._n_expected_tokenization_dims
        assert base_tokenization.shape[0] == 1
        assert len(base_tokenization[leading_dim_inds].shape) == 1
        assert base_tokenization[leading_dim_inds][0] == self.alphabet.cls_idx

        # Get the indices we wish to update
        inds_to_update_list.append(target_positions.to(self.device))
        inds_to_update_tuple = tuple(inds_to_update_list)

        # Build the masked tokenization for the target indices to update
        assert torch.equal(  # pylint: disable=no-member
            base_tokenization[inds_to_update_tuple],
            torch.tensor(  # pylint: disable=no-member
                [self.alphabet.tok_to_idx[parent_aa] for parent_aa in parent_aas]
            ),
        )
        base_tokenization[inds_to_update_tuple] = self.alphabet.mask_idx

        # Confirm that the base tokenization is now appropriately masked
        update_tuple_len = len(inds_to_update_tuple)
        expected_update_tuple_len = 3 if self._is_msa_transformer else 2
        assert update_tuple_len == expected_update_tuple_len
        assert all(isinstance(member, int) for member in inds_to_update_tuple[:-1])
        assert isinstance(inds_to_update_tuple[-1], torch.Tensor)
        self._check_masking(
            og_tokenization=og_tokenization,
            base_tokenization=base_tokenization,
            inds_to_update=inds_to_update_tuple,  # type: ignore
            parent_aas=parent_aas,
        )

        # Run our base tokenization through the model.
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            res = self.model(base_tokenization.to(self.device))

        # Get the logits of the target positions sequence
        expected_shape = (
            len(target_positions),
            len(self.alphabet.tok_to_idx),
        )
        target_logits = res["logits"][inds_to_update_tuple]
        assert target_logits.shape == expected_shape

        # Get log probs
        target_log_probs = F.log_softmax(target_logits, dim=-1).detach().cpu().numpy()
        assert target_log_probs.shape == expected_shape

        return target_log_probs

    def _check_masking(
        self,
        og_tokenization: torch.Tensor,
        base_tokenization: torch.Tensor,
        inds_to_update: Union[tuple[int, int, torch.Tensor], tuple[int, torch.Tensor]],
        parent_aas: tuple[str, ...],
    ) -> None:
        """Confirms that masking was appropriately performed. This is called within
        `self._get_masked_log_probs` as a sanity check. Specifically, this method
        confirms that...

        1)  Tokenization shape is not changed during masking.
        2)  The indices that are masked correctly map to the expected parent amino
            acid tokens before masking occurs.
        3)  The indices that are supposed to be masked are indeed masked.

        Args:
            og_tokenization (torch.Tensor): The tokenization of a parent protein
                sequence or processed MSA depending on whether an MSA or regular
                transformer is being used. No masking has been applied to this
                tensor.
            base_tokenization (torch.Tensor): The tokenization of a parent protein
                sequence or processed MSA depending on whether an MSA or regular
                transformer is being used. The positions given in `inds_to_update`
                are expected to be masked.
            inds_to_update (Union[tuple[int, int, torch.Tensor], tuple[int, torch.Tensor]]):
                A tuple that is either (0, positions) or (0, 0, positions), where
                `positions` are the positions that are to be masked and are given
                by a LongTensor. The 2-member tuple is used for indexing
                `base_tokenization` when a non-MSA transformer is being used and
                the 3-member tuple is used for indexing when an MSA transformer
                is being used.
            parent_aas (tuple[str, ...]): The expected amino acid identities at
                the positions that are masked.
        """

        # The og tokenization should match the base tokenization shape
        assert og_tokenization.shape == base_tokenization.shape

        # If we extract the target positions from the og tokenization, it should
        # be the parent amino acids
        extracted_parent_inds = og_tokenization[inds_to_update]
        idx_to_tok = {idx: tok for tok, idx in self.alphabet.tok_to_idx.items()}
        assert parent_aas == tuple(
            idx_to_tok[idx.item()] for idx in extracted_parent_inds
        )

        # If we extract the target positions from the base tokenization, everything
        # should be masked
        extracted_masked_inds = base_tokenization[inds_to_update]
        assert all(
            idx.item() == self.alphabet.mask_idx for idx in extracted_masked_inds
        )

    def _update_target_positions(
        self,
        target_positions: IntArray,
        parent: Union[str, MSA],
    ) -> torch.Tensor:
        """Used to adjust indices to account for the addition of a cls token as
        well as any insertions or deletions in an MSA.

        Args:
            target_positions (IntArray): The original target indices (0-indexed)
                as a numpy array.
            parent (Union[str, MSA]): Either a parent sequence or a
                parent MSA.

        Raises:
            ValueError: Raised if any of the indices in `target_positions` are
                not applicable. This occurs if the indices are out of range (i.e.
                longer than the sequence) or have been deleted from the MSA (i.e.
                due to a gap/insertion).

        Returns:
            torch.Tensor: A torch tensor giving the new positions.
        """

        # If we are an msa transformer, we need to use the `old_to_new` dictionary
        # to identify our new positions of interest
        if isinstance(parent, MSA):

            # Make sure every ind is in the target positions. If not, raise an error.
            if not all(ind in parent.old_to_new for ind in target_positions):
                raise ValueError(
                    "Not all requested target positions remain in the MSA after "
                    "it is processed. You have either mis-specified your target "
                    "positions or you need to relax restrictions on your MSA search."
                )

            new_positions = torch.tensor(  # pylint: disable=no-member
                [parent.old_to_new[ind] for ind in target_positions],
                dtype=torch.long,  # pylint: disable=no-member
            )
        else:
            new_positions = torch.tensor(  # pylint: disable=no-member
                target_positions, dtype=torch.long  # pylint: disable=no-member
            )

        # Add 1 for the cls token, which we assert exists in other functions
        return new_positions + 1

    def _tokenize(self, sequences: Union[SequenceList, list[MsaType]]) -> torch.Tensor:
        """Used to tokenize protein sequences or MSAs.

        Args:
            sequences (Union[SequenceList, list[MsaType]]): Either a list of
                sequences or a list of MSAs that will be tokenized.

        Returns:
            torch.Tensor: Either a 3D or 4D tensor giving the tokenized sequences
                or MSAs, respectively.
        """
        if self._is_msa_transformer:
            _, _, tokens = self.batch_converter(sequences)
        else:
            _, _, tokens = self.batch_converter(
                [(str(seqind), seq) for seqind, seq in enumerate(sequences)]
            )

        return tokens


class EsmPredictor(AbstractEsmPredictor):
    """A class that contains all methods needed for making zero-shot predictions
    using non-MSA transformer ESM models.
    """

    # Define class variables
    _allowed_models: dict[str, tuple[Callable, int, int]] = {
        "esm1_t34_670M_UR50S": esm.pretrained.esm1_t34_670M_UR50S,
        "esm1_t34_670M_UR50D": esm.pretrained.esm1_t34_670M_UR50D,
        "esm1_t34_670M_UR100": esm.pretrained.esm1_t34_670M_UR100,
        "esm1_t12_85M_UR50S": esm.pretrained.esm1_t12_85M_UR50S,
        "esm1_t6_43M_UR50S": esm.pretrained.esm1_t6_43M_UR50S,
        "esm1b_t33_650M_UR50S": esm.pretrained.esm1b_t33_650M_UR50S,
        "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
        "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
        "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
        "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
        "esm2_t36_3B_UR50D": esm.pretrained.esm2_t36_3B_UR50D,
        "esm2_t48_15B_UR50D": esm.pretrained.esm2_t48_15B_UR50D,
        "esm1v_t33_650M_UR90S_1": esm.pretrained.esm1v_t33_650M_UR90S_1,
        "esm1v_t33_650M_UR90S_2": esm.pretrained.esm1v_t33_650M_UR90S_2,
        "esm1v_t33_650M_UR90S_3": esm.pretrained.esm1v_t33_650M_UR90S_3,
        "esm1v_t33_650M_UR90S_4": esm.pretrained.esm1v_t33_650M_UR90S_4,
        "esm1v_t33_650M_UR90S_5": esm.pretrained.esm1v_t33_650M_UR90S_5,
    }

    # How many dimensions we expect a tokenized batch to have
    _n_expected_tokenization_dims: int = 2

    # pylint: disable=unused-argument, missing-function-docstring, invalid-name
    # Classmethods that are shortcuts to the various models
    @classmethod
    @_load_esm_model
    def esm1_t34_670M_UR50S(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    @classmethod
    @_load_esm_model
    def esm1_t34_670M_UR50D(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    @classmethod
    @_load_esm_model
    def esm1_t34_670M_UR100(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    @classmethod
    @_load_esm_model
    def esm1_t12_85M_UR50S(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    @classmethod
    @_load_esm_model
    def esm1_t6_43M_UR50S(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    @classmethod
    @_load_esm_model
    def esm1b_t33_650M_UR50S(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    @classmethod
    @_load_esm_model
    def esm2_t6_8M_UR50D(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    @classmethod
    @_load_esm_model
    def esm2_t12_35M_UR50D(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    @classmethod
    @_load_esm_model
    def esm2_t30_150M_UR50D(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    @classmethod
    @_load_esm_model
    def esm2_t33_650M_UR50D(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    @classmethod
    @_load_esm_model
    def esm2_t36_3B_UR50D(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    @classmethod
    @_load_esm_model
    def esm2_t48_15B_UR50D(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    @classmethod
    @_load_esm_model
    def esm1v_t33_650M_UR90S_1(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    @classmethod
    @_load_esm_model
    def esm1v_t33_650M_UR90S_2(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    @classmethod
    @_load_esm_model
    def esm1v_t33_650M_UR90S_3(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    @classmethod
    @_load_esm_model
    def esm1v_t33_650M_UR90S_4(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    @classmethod
    @_load_esm_model
    def esm1v_t33_650M_UR90S_5(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    # pylint: enable=unused-argument, missing-function-docstring, invalid-name


class EsmMsaPredictor(AbstractEsmPredictor):
    """A class that contains all methods needed for making zero-shot predictions
    using MSA transformer ESM models.
    """

    _allowed_models = {
        "esm_msa1_t12_100M_UR50S": esm.pretrained.esm_msa1_t12_100M_UR50S,
        "esm_msa1b_t12_100M_UR50S": esm.pretrained.esm_msa1b_t12_100M_UR50S,
    }
    # How many dimensions we expect a tokenized batch to have
    _n_expected_tokenization_dims = 3

    # Are we an MSA trasformer?
    _is_msa_transformer = True

    # pylint: disable=unused-argument, missing-function-docstring, invalid-name
    # Classmethod shortcuts
    @classmethod
    @_load_esm_model
    def esm_msa1_t12_100M_UR50S(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    @classmethod
    @_load_esm_model
    def esm_msa1b_t12_100M_UR50S(
        cls, pretrained: bool = True, device: str = DEFAULT_DEVICE
    ): ...

    # pylint: enable=unused-argument, missing-function-docstring, invalid-name
