import numpy
from typing import List

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from sklearn import mixture


def read_fasta(path: str) -> List[SeqRecord]:
    """
    Helper function to read FASTA file.
    :param path: path to a valid FASTA file
    :return: a list of SeqRecord objects.
    """
    try:
        return list(SeqIO.parse(path, "fasta"))
    except FileNotFoundError:
        raise  # Already says "No such file or directory"
    except Exception as e:
        raise ValueError(f"Could not parse '{path}'. Are you sure this is a valid fasta file?") from e
    pass


## Copied from https://colab.research.google.com/github/google-research/google-research/blob/master/aav/model_and_dataset_analysis/data_prep.ipynb#scrollTo=KXnqP29_sAXj
# Covariance type to use in Gaussian Mixture Model.
_COVAR_TYPE = 'full'
# Number of components to use in Gaussian Mixture Model.
_NUM_COMPONENTS = 2

class TwoGaussianMixtureModelLabeler(object):
  """Emits class labels from Gaussian Mixture given input data.

  Input data is encoded as 1-D arrays.  Allows for an optional ambiguous label
  between the two modelled Gaussian distributions. Without the optional
  ambigouous category, the two labels are:
     0 - For values more likely derived from the Gaussian with smaller mean
     2 - For values more likely derived from the Gaussian with larger mean

  When allowing for an ambiguous category the three labels are:
     0 - For values more likely derived from the Gaussian with smaller mean
     1 - For values which fall within an ambiguous probability cutoff.
     2 - For values more likely derived from the Gaussian with larger mean
  """

  def __init__(self, data):
    """Constructor.

    Args:
      data: (numpy.ndarray or list) Input data to model with Gaussian Mixture.
      Input data is presumed to be in the form [x1, x2, ...., xn].
    """
    self._data = numpy.array([data]).T
    self._gmm = mixture.GaussianMixture(
        n_components=_NUM_COMPONENTS,
        covariance_type=_COVAR_TYPE).fit(self._data)

    # Re-map the gaussian with smaller mean to the "0" label.
    self._label_by_index = dict(
        list(zip([0, 1],
                 numpy.argsort(self._gmm.means_[:, 0]).tolist())))
    self._label_by_index_fn = numpy.vectorize(lambda x: self._label_by_index[x])

  def predict(self, values, probability_cutoff=0.):
    """Provides model labels for input value(s) using the GMM.

    Args:
      values: (array or single float value) Value(s) to infer a label on.
        When values=None, predictions are run on self._data.
      probability_cutoff: (float) Proability between 0 and 1 to identify which
        values correspond to ambiguous labels.  At probablity_cutoff=0 (default)
        it only returns the original two state predictions.

    Returns:
      A numpy array with length len(values) and labels corresponding to 0,1 if
      probability_cutoff = 0 and 0, 1, 2 otherwise.  In the latter, 0
      corresponds to the gaussian with smaller mean, 1 corresponds to the
      ambiguous label, and 2 corresponds to the gaussian with larger mean.
    """
    values = numpy.atleast_1d(values)
    values = numpy.array([values]).T
    predictions = self._label_by_index_fn(self._gmm.predict(values))
    # Re-map the initial 0,1 predictions to 0,2.
    predictions *= 2
    if probability_cutoff > 0:
      probas = self._gmm.predict_proba(values)
      max_probas = numpy.max(probas, axis=1)
      ambiguous_values = max_probas < probability_cutoff

      # Set ambiguous label as 1.
      predictions[ambiguous_values] = 1
    return predictions