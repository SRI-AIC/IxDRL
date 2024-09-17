import logging
import numpy as np
from typing import List, Optional
from ixdrl import Data, CategoricalDistribution, MultiCategoricalDistribution, NormalDistribution, \
    MultiData, AtomicData, Rollout, Rollouts
from ixdrl.analysis import AnalysisDimensionBase, RolloutsAnalyses, RolloutAnalysis
from ixdrl.util.math import mean_pairwise_distances, jensen_shannon_divergence, jensen_renyi_divergence

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

ATOMIC_DIST_METRIC = 'cosine'


class FamiliarityAnalysis(AnalysisDimensionBase):
    """
    This analysis captures the agent’s familiarity with situations by estimating the epistemic uncertainty,
    corresponding to the subjective uncertainty, i.e., due to limited data or lack of experience with the environment.
    We follow approaches in the (model-based) RL literature where epistemic uncertainty is measured by estimating the
    level of disagreement between different predictive models, forming an ensemble, that are trained independently,
    usually by random sub-sampling of a common replay buffer. The output values are in the range [-1,1].
    """

    def __init__(self, num_processes: int = 1):
        """
        Creates a new familiarity analysis.
        :param int num_processes: the number of parallel processes to use for this analysis. A value of `-1` or `None`
        will use all available cpus.
        """
        super().__init__('Familiarity', num_processes)

    def analyze(self, rollouts: Rollouts) -> Optional[RolloutsAnalyses]:

        # perform analysis over all rollouts
        all_analyses = self._analyze_parallel(self._get_familiarity, rollouts)
        if all_analyses is None:
            logging.error('Cannot compute familiarity: no valid ensemble data found in values, action values, '
                          'predicted observations or predicted rewards in the datapoints.')
            return None  # no data available in the datapoints

        return all_analyses

    def _get_familiarity(self, rollout: Rollout) -> Optional[RolloutAnalysis]:

        def _ensemble_familiarity(data: List[Data]) -> Optional[np.ndarray]:
            # first check data
            if len(data) == 0 or any(pred is None for pred in data):
                return None  # invalid data

            p_type = type(data[0])
            if any(type(pred) is not p_type for pred in data):
                return None  # incompatible data types

            d = None  # to put the mean distances/divergences across the models in the ensemble

            # check data type and get familiarity accordingly over the ensemble
            if p_type is AtomicData:
                # if point predictions, compute familiarity from cosine distance between predictions
                # each shape: (batch, *, n_dims), length: ens_size
                preds = np.stack([pred.astype(np.float64)
                                  for pred in data], axis=1)  # shape: (batch, *, ens_size, n_dims)
                dists = mean_pairwise_distances(preds, metric=ATOMIC_DIST_METRIC)  # shape: (bath, *)
                dists = np.mean(dists, axis=tuple(np.arange(1, len(dists.shape))))  # shape: (batch, )
                d = dists

            elif p_type is CategoricalDistribution:
                # if categorical, compute familiarity from JSD between predicted distributions
                # each shape: (batch, *, n_dims, n_atoms), length: ens_size
                dists = np.stack([dist.probs.astype(np.float64)
                                  for dist in data], axis=1)  # (batch, *, ens_size, n_dims, n_atoms)
                dists = np.swapaxes(dists, -2, -3)  # shape: (batch, *, n_dims, ens_size, n_atoms)
                divs = jensen_shannon_divergence(dists)  # shape: (batch, *, n_dims)
                divs = np.mean(divs, axis=tuple(np.arange(1, len(divs.shape))))  # shape: (batch, )
                d = divs

            elif p_type is NormalDistribution:
                # if normal, compute familiarity from JRD between predicted distributions
                # each shape: (batch, *, n_dims), length: ens_size
                mu = np.stack([d.mean.astype(np.float64) for d in data], axis=1)  # (batch, *, ens_size, n_dims)
                sigma = np.stack([d.std ** 2 for d in data], axis=1)  # (batch, *, ens_size, n_dims)
                divs = jensen_renyi_divergence(mu, sigma, clip=True)  # shape: (batch, *)
                divs = np.mean(divs, axis=tuple(np.arange(1, len(divs.shape))))  # shape: (batch, )
                d = divs

            # F(t) = 1 - 2 mean( d(f_i, f_j) )
            return None if d is None else 1. - 2. * d

        def __get_familiarity(data: Data) -> Optional[np.ndarray]:
            # checks if data represents output of models in an ensemble, compute familiarity
            if data is None:
                return None  # no data
            if isinstance(data, MultiCategoricalDistribution):
                return _ensemble_familiarity(data.dists)
            if isinstance(data, MultiData):
                return _ensemble_familiarity(data.data)
            return None  # data type not supported, has to represent ensemble

        # tries to get familiarity from different data
        familiarity = None
        if rollout.data.value is not None:
            familiarity = __get_familiarity(rollout.data.value)
        if familiarity is None and rollout.data.action_values is not None:
            familiarity = __get_familiarity(rollout.data.action_values)
        if familiarity is None and rollout.data.pred_obs is not None:
            familiarity = __get_familiarity(rollout.data.pred_obs)
        if familiarity is None and rollout.data.pred_rwd is not None:
            familiarity = __get_familiarity(rollout.data.pred_rwd)

        if familiarity is None:
            return None  # no data available in the datapoints

        # select timesteps and return dictionary with single entry
        return {self.name: familiarity}  # final shape: (batch, )
