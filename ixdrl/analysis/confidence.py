import logging
import numpy as np
from typing import Optional, get_args
from collections import OrderedDict
from ixdrl import CategoricalDistribution, NormalDistribution, MultiCategoricalDistribution, Data, \
    MultiData, Distribution, Rollout, get_num_action_dimensions, Rollouts
from ixdrl.analysis import AnalysisDimensionBase, RolloutsAnalyses, RolloutAnalysis
from ixdrl.util.math import evenness_index, gaussian_entropy_dispersion, gaussian_variation_coefficient

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class ConfidenceAnalysis(AnalysisDimensionBase):
    """
    This analysis characterizes how confident an agent is with regards to action-selection at each timestep.
    The purpose of this dimension is to denote situations in which the agent is (un)certain of what to do and provide
    good opportunities to request guidance by a human user or identify tasks in which the agent would require more
    training.
    Confidence is calculated from the Pielou’s evenness index which corresponds to the normalized entropy of a
    distribution.
    """

    def __init__(self, num_processes: int = 1):
        """
        Creates a new confidence analysis.
        :param int num_processes: the number of parallel processes to use for this analysis. A value of `-1` or `None`
        will use all available cpus.
        """

        super().__init__('Confidence', num_processes)

    def analyze(self, rollouts: Rollouts) -> Optional[RolloutsAnalyses]:

        # perform analysis over all rollouts
        all_analyses = self._analyze_parallel(self._get_confidence, rollouts)
        if all_analyses is None:
            logging.error(f'Cannot compute confidence, data type not supported or incompatible data.')
            return None  # cannot extract confidence from data

        return all_analyses

    def _get_confidence(self, rollout: Rollout) -> Optional[RolloutAnalysis]:

        # checks data
        if rollout.data.action_dist is None:
            return None

        def _confidence_discrete(dist: np.ndarray):
            # confidence: C(s|π)=1-2J(π(⋅│s))
            return 1. - 2. * evenness_index(dist.astype(np.float64), axis=-1)  # shape: (batch, )

        def _confidence_continuous(dist: Distribution):
            if isinstance(dist, NormalDistribution):
                # confidence: C(t)= 1 - 2 c_v(π(s_t )), with c_v the coefficient of variation (CV)
                mean = dist.mean.astype(np.float64)  # shape: (batch, n_action_dims)
                std = dist.std.astype(np.float64)  # shape: (batch, n_action_dims)
                conf = 1. - 2. * gaussian_entropy_dispersion(mean, std, clip=True)  # shape: (batch, n_action_dims)
                conf = conf.swapaxes(0, 1)  # shape: (*act_dims, batch)
                return conf.reshape(-1, conf.shape[-1])  # shape: (act_dims, batch)
                # TODO might be better consider multivariate Gaussians and use CV, but we lose confidence per act dim..
                # conf = 1. - 2. * gaussian_variation_coefficient(mean, std ** 2, clip=True)  # shape: (batch, )
                # return [conf]  # shape: (batch, )

            return None  # currently not supported continuous distribution

        def __get_confidence(data: Data, target_dims: int):
            # check type of action distribution, organize data accordingly
            if isinstance(data, CategoricalDistribution):
                # if categorical, shape: (batch, n_actions), compute confidence for selected batch indices
                return [_confidence_discrete(data.probs)]  # shape (batch, ), length: 1
            if isinstance(data, MultiCategoricalDistribution) and len(data.dists) == target_dims:
                # if multi-categorical, each shape: (batch, n_actions), select from each dimension
                # each shape: (batch, ), length: n_dim_actions
                return [_confidence_discrete(dist.probs) for dist in data.dists]
            if isinstance(data, MultiData) and len(data.data) == target_dims:
                # if multi-data, each dimension has own distribution, so compute confidence for each
                return [__get_confidence(dist.probs, 1) for dist in data.data]
            if isinstance(data, get_args(Distribution)):
                # assume continuous dist
                return _confidence_continuous(data)
            return None  # no compatible data

        # try to get confidence for
        confidences = __get_confidence(rollout.data.action_dist, len(rollout.action_labels))
        if confidences is None:
            return None

        # create dictionary containing confidence for each timestep, organized by action dimension
        conf_dict = OrderedDict({self.name: np.nanmean(confidences, axis=0)})  # shape: (batch, )
        n_act_dims = get_num_action_dimensions(rollout.action_space)
        if 1 < n_act_dims == len(confidences):
            if n_act_dims == len(rollout.action_labels):
                labels = rollout.action_labels
            else:
                labels = [f'Action Dim {i}' for i in range(len(rollout.action_space.shape))]
            for i in range(n_act_dims):
                conf_dict[f'{self.name}-{labels[i]}'] = confidences[i]  # each shape: (batch, )

        # add also mean value
        return conf_dict
