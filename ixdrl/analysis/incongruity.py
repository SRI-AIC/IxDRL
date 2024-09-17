import logging
import numpy as np
from typing import Optional
from ixdrl import MultiCategoricalDistribution, Data, AtomicData, CategoricalDistribution, \
    NormalDistribution, MultiData, Rollout, Rollouts
from ixdrl.analysis import AnalysisDimensionBase, RolloutsAnalyses, RolloutAnalysis

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class IncongruityAnalysis(AnalysisDimensionBase):
    """
    Captures situations in which there are internal inconsistencies between the expected value of a situation.
    We capture incongruity via temporal difference (TD) error at each step, i.e., :math:`r_(t+1)+γV(s_(t+1) )-V(s_t)`.
    This captures the difference between expected state value and the on-policy target. This dimension might denote
    situations where learning has not yet converged or situations in which the reward received by the agent is very
    stochastic (or unexpected w.r.t. training), hence the agent’s predictions are not very accurate.
    """

    def __init__(self, num_processes: int = 1):
        """
        Creates a new incongruity analysis.
        :param int num_processes: the number of parallel processes to use for this analysis. A value of `-1` or `None`
        will use all available cpus.
        """
        super().__init__('Incongruity', num_processes)

    def analyze(self, rollouts: Rollouts) -> Optional[RolloutsAnalyses]:

        # perform analysis over all rollouts
        all_analyses = self._analyze_parallel(self._get_incongruity, rollouts)
        if all_analyses is None:
            logging.error('Cannot compute goal conduciveness, no discount factor.value or reward data available in the'
                          'datapoints, or invalid data types found.')
            return None  # cannot extract confidence from data

        # gets reward range to normalize incongruity
        rwd_range = np.inf  # np.abs(np.diff(next(iter(rollouts.values())).rwd_range))[0]  # assumes all envs/tasks the same
        if np.isinf(rwd_range):
            # if not defined, get range across all episodes
            all_rwds = np.concatenate([rollout.data.reward.flatten() for rollout in rollouts.values()])
            rwd_range = np.ptp(all_rwds)
            if rwd_range == 0:
                # if constant reward, assume min is 0
                rwd_range = np.max(all_rwds)

        # normalize incongruity and clip values across data
        for dim, analyses in all_analyses.items():
            for rollout_id, analysis in analyses.items():
                analysis /= rwd_range
                analysis = np.clip(analysis, -1, 1)
                analyses[rollout_id] = analysis

        return all_analyses

    def _get_incongruity(self, rollout: Rollout) -> Optional[RolloutAnalysis]:

        # check data
        if rollout.discount is None or rollout.data.value is None or rollout.data.reward is None:
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
        values = _get_values(rollout.data.value)
        if values is None:
            return None

        rewards = rollout.data.reward.astype(np.float64).flatten()  # shape: (batch, )

        # incongruity I(t) = r_t+γV(s_t) - V(s_(t-1) ) = [R(s_(t-1), a_(t-1))+γV(s_t)] - V(s_(t-1) )
        targets = rewards[:-1] + rollout.discount * values[1:]
        incongruity = np.full_like(values, np.nan)
        incongruity[1:] = targets - values[:-1]  # undefined in the first timestep since we don't have R and V of t=-1

        return {self.name: incongruity}
