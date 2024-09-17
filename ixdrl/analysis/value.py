import logging
import numpy as np
from typing import Optional
from ixdrl import MultiCategoricalDistribution, Data, AtomicData, CategoricalDistribution, \
    NormalDistribution, MultiData, Rollout, Rollouts
from ixdrl.analysis import AnalysisDimensionBase, RolloutsAnalyses, RolloutAnalysis

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class ValueAnalysis(AnalysisDimensionBase):
    """
    Characterizes the long-term importance of a situation as ascribed by the agent’s value function, which estimates
    the cumulative discounted reward of being in a state and executing the agent’s policy thereafter. Given that the
    absolute value of the MDP’s reward function is bounded by parameter $R_max$, we compute value by normalizing the
    value function against $R_max / (1-\\gamma)$.
    """

    def __init__(self, num_processes: int = 1):
        """
        Creates a new value analysis.
        :param int num_processes: the number of parallel processes to use for this analysis. A value of `-1` or `None`
        will use all available cpus.
        """
        super().__init__('Value', num_processes)

    def analyze(self, rollouts: Rollouts) -> Optional[RolloutsAnalyses]:

        # perform analysis over all rollouts
        all_analyses = self._analyze_parallel(self._get_value, rollouts)
        if all_analyses is None:
            logging.error('Cannot compute goal conduciveness, no discount factor.value or reward data available in the'
                          'datapoints, or invalid data types found.')
            return None  # cannot extract confidence from data

        # normalize value and clip values across data
        for dim, analyses in all_analyses.items():
            values = np.concatenate([value.flatten() for value in analyses.values()])
            _minV = np.min(values)
            _maxV = np.max(values)
            for rollout_id, analysis in analyses.items():
                analysis = (analysis - _minV) / (_maxV - _minV)  # normalize values in [0,1]
                analysis = 2 * analysis - 1  # V(t) = 2 V - 1
                analyses[rollout_id] = analysis

        return all_analyses

    def _get_value(self, rollout: Rollout) -> Optional[RolloutAnalysis]:

        # check data
        if rollout.data.value is None:
            return None

        def _get_values(data: Data) -> Optional[np.ndarray]:
            if isinstance(data, AtomicData) and 0 < len(data.shape) <= 2 and data.shape[-1] == 1:
                # if atomic, we already have one value per step, shape: (batch, 1)
                if len(data.shape) == 2:
                    return data.astype(np.float64).flatten()  # flatten, shape: (batch, )
            if isinstance(data, CategoricalDistribution):
                # if categorical, shape is (batch, n_atoms), so take mean value
                return np.sum(data.support.astype(np.float64) * data.probs.astype(np.float64), axis=-1)  # shape: (batch, )
            if isinstance(data, NormalDistribution) and 0 < len(data.mean.shape) <= 2 and data.mean.shape[-1] == 1:
                # if normal, just return means
                return data.mean.astype(np.float64)  # shape: (batch, )
            if isinstance(data, MultiCategoricalDistribution):
                # if multi-categorical, then probably comes from ensemble of models, get mean of those
                return np.mean([_get_values(dist) for dist in data.dists], axis=0)  # shape: (batch, )
            if isinstance(data, MultiData):
                # if multi-data, then probably comes from ensemble of models, get mean of those
                return np.mean([_get_values(dist) for dist in data.data], axis=0)  # shape: (batch, )
            return None  # unsupported data type

        # get value (mean) data
        value = _get_values(rollout.data.value)
        if value is None:
            return None

        return {self.name: value}
