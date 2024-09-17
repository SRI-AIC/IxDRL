import logging
import numpy as np
from typing import List, Optional
from collections import OrderedDict
from ixdrl import AtomicData, CategoricalDistribution, MultiCategoricalDistribution, Data, MultiData, \
    Rollout, get_num_action_dimensions, Rollouts
from ixdrl.analysis import AnalysisDimensionBase, RolloutsAnalyses, RolloutAnalysis

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class RiskinessAnalysis(AnalysisDimensionBase):
    """
    This analysis tries to identify “edge cases,” where the impact of performing the “right” action vs. the “wrong” one
    is high. If action value data is available, we compute the riskiness associated with a timestep from the maximal
    value difference between any two actions. If action value data is not available but action distribution data is, we
    compute riskiness from the maximal difference between the probability of executing any two actions. Otherwise,
    if the action selection probability is available, we use it to compute riskiness.
    """

    def __init__(self, num_processes: int = 1):
        """
        Creates a new riskiness analysis.
        :param int num_processes: the number of parallel processes to use for this analysis. A value of `-1` or `None`
        will use all available cpus.
        """
        super().__init__('Riskiness', num_processes)

    def analyze(self, rollouts: Rollouts) -> Optional[RolloutsAnalyses]:
        # perform analysis over all rollouts
        all_analyses = self._analyze_parallel(self._get_riskiness, rollouts)
        if all_analyses is None:
            logging.error('Cannot compute riskiness: no valid action values, distribution or action probability data '
                          'available in the datapoints.')
            return None  # no valid data

        _minQ = np.inf
        _maxQ = -np.inf

        # checks whether analyses data is from action values, in which case needs to be normalized
        action_values = next(iter(rollouts.values())).data.action_values
        if action_values is not None and isinstance(
                action_values, (AtomicData, MultiData, MultiCategoricalDistribution, CategoricalDistribution)):
            all_values = np.concatenate([analysis
                                         for analyses in all_analyses.values()
                                         for analysis in analyses.values()])
            _minQ = np.min(all_values)
            _maxQ = np.max(all_values)

        # compute riskiness across data
        for dim, analyses in all_analyses.items():
            for rollout_id, analysis in analyses.items():
                if not np.isinf(_minQ):
                    analysis = (analysis - _minQ) / (_maxQ - _minQ)  # normalize values
                if len(analysis.shape) > 1 and analysis.shape[1] > 1:
                    analysis = np.ptp(analysis, axis=-1)  # compute value/prob range
                else:
                    analysis = analysis.flatten()

                # R(t) = 2 Range - 1
                analysis = 2 * analysis - 1
                analyses[rollout_id] = analysis

        # check mean data
        if len(all_analyses) > 1:
            mean_data = {rollout_id: np.nanmean([analyses[rollout_id] for analyses in all_analyses.values()], axis=0)
                         for rollout_id in next(iter(all_analyses.values())).keys()}
            all_analyses = OrderedDict({self.name: mean_data, **all_analyses})

        return all_analyses

    def _get_riskiness(self, rollout: Rollout) -> Optional[RolloutAnalysis]:

        def _get_action_data_per_dim(data: Data, is_action_value: bool) -> Optional[List[np.ndarray]]:
            if isinstance(data, AtomicData):
                # if atomic, then we already have one value per action (single dimension)
                return [data.astype(np.float64).reshape((data.shape[0], -1))]  # shape: (batch, n_actions)

            if isinstance(data, MultiData) and len(data.data) == len(rollout.action_labels):
                # if multi-data, assume one value for each dimension
                return [_get_action_data_per_dim(act_data, is_action_value)[0]
                        for act_data in data.data]  # shape: (batch, n_actions), length: act_dims

            if (isinstance(data, MultiCategoricalDistribution) and
                    len(data.dists) == len(rollout.action_labels)):
                # if multi-categorical, we have to get the mean values for each action, for each dimension
                return [_get_action_data_per_dim(dist, is_action_value)[0]
                        for dist in data.dists]  # shape: (batch, n_actions), length: act_dims

            if isinstance(data, CategoricalDistribution):
                if is_action_value:
                    # if categorical and value data, we have to get the mean values for each action (single dimension)
                    action_val_dists = data.support.astype(np.float64) * \
                                       data.probs.astype(np.float64)  # shape: (batch, n_actions, n_atoms)
                    return [np.sum(action_val_dists, axis=-1)]  # shape: (batch, n_actions), length: 1
                else:
                    # otherwise its action probabilities, so return the probs directly
                    return [data.probs.astype(np.float64)]  # shape: (batch, n_actions), length: 1

            return None

        act_data = None
        if rollout.data.action_values is not None:
            action_values = _get_action_data_per_dim(rollout.data.action_values, is_action_value=True)
            if action_values is not None:
                # if action values available, R(t) = 2(max(a_1)Q(s_t,a_1) - min (a_2)Q(s_t,a_2 )) - 1
                act_data = action_values  # shape (act_dims, batch, n_actions)

        if act_data is None and rollout.data.action_dist is not None:
            action_dist = _get_action_data_per_dim(rollout.data.action_dist, is_action_value=False)
            if action_dist is not None:
                # if action probabilities available, R(t) = 2(max(a_1)π(s_t,a_1 )-min(a_2)π(s_t,a_2 )) - 1
                act_data = action_dist  # shape (act_dims, batch, n_actions)

        if act_data is None and rollout.data.action_prob is not None:
            # if only selected action prob available, then R(t) = 2 P(a_t│π) - 1
            action_probs = _get_action_data_per_dim(rollout.data.action_prob, is_action_value=False)
            if action_probs is not None:
                act_data = action_probs  # shape: (act_dims, batch, 1)

        if act_data is None:
            return None  # no data available in the datapoints

        # create dictionary containing riskiness for each timestep, organized by action dimension
        risk_dict = OrderedDict()
        n_act_dims = get_num_action_dimensions(rollout.action_space)
        if 1 < n_act_dims == len(act_data):
            if n_act_dims == len(rollout.action_labels):
                labels = rollout.action_labels
            else:
                labels = [f'Action Dim {i}' for i in range(len(rollout.action_space.shape))]
            for i in range(n_act_dims):
                risk_dict[f'{self.name}-{labels[i]}'] = act_data[i]  # each shape: (batch, n_actions)
        else:
            risk_dict = {self.name: act_data[0]}  # shape: (batch, n_actions)

        return risk_dict
