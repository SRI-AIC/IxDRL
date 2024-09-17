import logging
import numpy as np
from collections import OrderedDict
from typing import List
from ixdrl import Rollouts
from ixdrl.analysis import AnalysisDimensionBase, RolloutsAnalyses
from ixdrl.analysis.value import ValueAnalysis
from ixdrl.analysis.confidence import ConfidenceAnalysis
from ixdrl.analysis.familiarity import FamiliarityAnalysis
from ixdrl.analysis.stochasticity import StochasticityAnalysis
from ixdrl.analysis.goal_conduciveness import GoalConducivenessAnalysis
from ixdrl.analysis.incongruity import IncongruityAnalysis
from ixdrl.analysis.riskiness import RiskinessAnalysis

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class FullAnalysis(object):
    """
    Represents a complete or full analysis of interestingness, i.e., containing all possible analyses that can be
    performed. This is mostly useful to organize all analyses and create separate directories in which to save reports.
    """

    def __init__(self, data: Rollouts,
                 derivative_accuracy: int = 4,
                 num_processes: int = -1):
        """
        Creates a full set of analyses.
        :param Rollouts data: the interaction data to be analyzed that was collected for different episodes.
        :param int derivative_accuracy: the accuracy used by finite difference methods. Needs to be a positive, even number.
        :param int num_processes: the number of parallel processes to use for this analysis. A value of `-1` or `None`
        will use all available cpus.
        """
        assert data is not None and len(data) > 0, 'Rollout data cannot be None or have zero length.'
        self.rollouts: Rollouts = data
        self.num_processes = num_processes

        # creates analyses set
        self.analyses: List[AnalysisDimensionBase] = [
            ValueAnalysis(num_processes),
            ConfidenceAnalysis(num_processes),
            GoalConducivenessAnalysis(derivative_accuracy, num_processes),
            RiskinessAnalysis(num_processes),
            IncongruityAnalysis(num_processes),
            StochasticityAnalysis(num_processes),
            FamiliarityAnalysis(num_processes)
        ]

    def __len__(self):
        return len(self.analyses)

    def __iter__(self):
        return iter(self.analyses)

    def analyze(self) -> RolloutsAnalyses:
        """
        Performs analysis of interestingness over all available rollout data using all analysis dimensions.
        :rtype: RolloutsAnalyses
        :return: a dictionary of the form `{dimension: {rollout_id: values}}` containing the interestingness values
        for each rollout (each an array of shape (timesteps, )), organized for each extracted dimension.
        """
        logging.info('===================================================================')
        logging.info(f'Analyzing interestingness using {len(self.analyses)} analyses over '
                     f'{len(self.rollouts)} rollouts...')
        logging.info('===================================================================')

        # iterates over rollouts and analyses
        results = OrderedDict({})
        for analysis in self.analyses:
            logging.info('___________________________________________________________________')
            logging.info(f'Running "{analysis.name}" analysis for {len(self.rollouts)} rollouts...')

            analysis_res = analysis.analyze(self.rollouts)
            if analysis_res is None:
                logging.info(f'\t"{analysis.name}" could not be computed')
                continue
            else:
                vals = np.concatenate(list(analysis_res[analysis.name].values()), axis=0)
                logging.info(f'\tMean "{analysis.name}": {np.nanmean(vals):.2f} ± {np.nanstd(vals):.2f}')

            results.update(analysis_res)

        return results
