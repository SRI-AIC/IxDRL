import logging
import numpy as np
from typing import Optional
from ixdrl import Data, CategoricalDistribution, MultiCategoricalDistribution, \
    NormalDistribution, MultiData, Rollout, Rollouts
from ixdrl.analysis import AnalysisDimensionBase, RolloutsAnalyses, RolloutAnalysis
from ixdrl.util.math import ordinal_dispersion, gaussian_variation_coefficient

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class StochasticityAnalysis(AnalysisDimensionBase):
    """
    This analysis captures the environment's aleatoric uncertainty. This is the statistical uncertainty representative
    of the inherent system stochasticity, i.e., the unknowns that differ each time the same experiment is run. Here, we
    capture the uncertainty around what happens when we execute the same action in the same state. This analysis
    requires an algorithm that models the uncertainty of the agent’s environment, i.e., an algorithm implementing
    distributional RL [Bellemare2017]. We then capture stochasticity using a measure of statistical dispersion of
    distributions. The output values are in the range [-1,1].
    """

    def __init__(self, num_processes: int = 1):
        """
        Creates a new environment stochasticity analysis.
        :param int num_processes: the number of parallel processes to use for this analysis. A value of `-1` or `None`
        will use all available cpus.
        """
        super().__init__('Stochasticity', num_processes)

    def analyze(self, rollouts: Rollouts) -> Optional[RolloutsAnalyses]:

        # perform analysis over all rollouts
        all_analyses = self._analyze_parallel(self._get_stochasticity, rollouts)
        if all_analyses is None:
            logging.error('Cannot compute environment stochasticity: no valid distributional data found in values, '
                          'action values, predicted observations or predicted rewards in the datapoints.')
            return None  # cannot extract confidence from data

        return all_analyses

    def _get_stochasticity(self, rollout: Rollout) -> Optional[RolloutAnalysis]:

        def __get_stochasticity(data: Data) -> Optional[np.ndarray]:
            if data is None:
                return None  # no data

            if isinstance(data, MultiCategoricalDistribution):
                # consider that each distribution is the output of a model in an ensemble, so get mean of those
                disps = [__get_stochasticity(dist) for dist in data.dists]  # shape: (batch, ), length: ensemble_size
                if any(disp is None for disp in disps):
                    return None
                return np.mean(disps, axis=0)  # shape: (batch, )

            if isinstance(data, MultiData):
                # consider that each distribution is the output of a model in an ensemble, so get mean of those
                disps = [__get_stochasticity(dist) for dist in data.data]  # shape: (batch, ), length: ensemble_size
                if any(disp is None for disp in disps):
                    return None
                return np.mean(disps, axis=0)  # shape: (batch, )

            if isinstance(data, CategoricalDistribution):
                # if categorical distribution over values, use Leik's dispersion, S_e(t) = 1 - 4 |D(Data(⋅│s_t ))-0.5|
                disp = ordinal_dispersion(data.probs.astype(np.float64), axis=-1)  # shape: (batch, *)
                disp = 1 - 4 * np.abs(disp - 0.5)
                disp = np.mean(disp, axis=tuple(range(1, len(disp.shape))))  # shape: (batch, )
                return disp

            if isinstance(data, NormalDistribution):
                # assume multivariate Gaussian, then S_e(t) = 2 c_v(Data(.|s_t )) - 1 (w/ Reyment's CV)
                mu = data.mean.astype(np.float64)
                sigma = data.std.astype(np.float64) ** 2
                diagonal = mu.shape == sigma.shape
                disps = 2. * gaussian_variation_coefficient(
                    mu, sigma, diagonal=diagonal, clip=True) - 1  # shape: (batch, *)
                # get mean over all components of the data, if more than 1
                return np.mean(disps, axis=tuple(np.arange(1, len(disps.shape))))  # shape: (batch, )

            return None  # distribution type not supported

        # tries to get stochasticity from different data
        stochasticity = None
        if rollout.data.value is not None:
            stochasticity = __get_stochasticity(rollout.data.value)
        if stochasticity is None and rollout.data.action_values is not None:
            stochasticity = __get_stochasticity(rollout.data.action_values)
        if stochasticity is None and rollout.data.pred_obs is not None:
            stochasticity = __get_stochasticity(rollout.data.pred_obs)
        if stochasticity is None and rollout.data.pred_rwd is not None:
            stochasticity = __get_stochasticity(rollout.data.pred_rwd)

        if stochasticity is None:
            return None  # no data available in the datapoints

        # select timesteps and return dictionary with single entry
        return {self.name: stochasticity}  # final shape: (batch, )
