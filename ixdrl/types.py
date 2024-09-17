from typing import Optional, NamedTuple, List, Union
import numpy as np

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

"""
Specifies all possible data types that can appear in an interaction datapoint.
"""


class CategoricalDistribution(object):
    """
    Represents a categorical probability distribution.
    """

    def __init__(self, probs: np.ndarray, support: Optional[np.ndarray] = None):
        """
        Creates a new categorical distribution.
        :param np.ndarray probs: the discrete probabilities. It is assumed that `np.sum(probs, axis=-1)` will return an
        array of 1.
        :param np.ndarray support: the distribution support i.e., the centers or labels of each "category,". If `None`,
        it is assumed that each category will have a integer label, from `0` to the size of `probs.shape[-1]`.
        """
        self.probs: np.ndarray = probs  # shape: (*, [data_shape,] num_categories)
        if support is None:
            support = np.zeros_like(probs, dtype=int)
            support[..., :] = np.arange(support.shape[-1])  # support are zero-based indices
        self.support: np.ndarray = support


class MultiCategoricalDistribution(NamedTuple):
    """
    Represents a multi categorical distribution, where each dimension is itself a standard categorical distribution.
    The number of categories can vary among categories.
    """
    dists: List[CategoricalDistribution]


class NormalDistribution(NamedTuple):
    """
    Represents a normal or Gaussian distribution, characterized by mean and std vectors.
    """
    mean: np.ndarray
    std: np.ndarray


class UniformDistribution(NamedTuple):
    """
    Represents a uniform distribution characterized by lower and upper bound vectors.
    """
    lower: np.ndarray
    upper: np.ndarray


AtomicData = np.ndarray  # meant to represent single predictions shape: (*, data_shape)
Distribution = Union[CategoricalDistribution, MultiCategoricalDistribution, NormalDistribution, UniformDistribution]


class MultiData(NamedTuple):
    """
    Represents multiple prediction data (like the output of ensemble models), whose predictions can themselves be
    points or distributions.
    """
    data: List[Union[AtomicData, Distribution]]


Data = Union[AtomicData, Distribution, MultiData]


def get_mean_data(data: Data) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Utility method to get the mean data, e.g., the mean prediction, of the given batch-formatted data.
    :param Data data: the data whose expected value we want to estimate.
    :rtype: np.ndarray or list(np.ndarray)
    :return: an array of shape (batch_size, *) containing the mean data value, or a list of means for multi-dimensional
    data.
    """
    if isinstance(data, AtomicData):
        return data  # single prediction, so already in correct shape (batch, *)
    if isinstance(data, MultiData):
        # if multiple predictions, gets mean of list of shapes (batch, *)
        return np.mean([get_mean_data(d) for d in data.data], axis=0, keepdims=False)
    if isinstance(data, CategoricalDistribution):
        return np.sum(data.support * data.probs, axis=-1, keepdims=True)  # gets mean via weighted average
    if isinstance(data, MultiCategoricalDistribution):
        return [get_mean_data(dist) for dist in data.dists]  # we have multiple means
    if isinstance(data, NormalDistribution):
        return data.mean  # simply return the mean param
    if isinstance(data, UniformDistribution):
        return 0.5 * (data.upper - data.lower)  # return mid point of distribution
