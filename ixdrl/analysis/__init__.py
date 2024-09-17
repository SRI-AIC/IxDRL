import logging
import os
import tqdm
import numpy as np
import pandas as pd
from abc import abstractmethod, ABC
from typing import List, Dict, Optional, Callable
from ixdrl import Rollout, Rollouts
from ixdrl.data_collection import MEAN_ROLLOUT_ID
from ixdrl.util.mp import run_parallel
from ixdrl.util.plot import dummy_plotly, plot_timeseries, plot_bar, plot_matrix, plot_radar

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

TIMESTEP_COL = 'Timestep'
ROLLOUT_ID_COL = 'Rollout'
SMOOTH_FACTOR = 0.9
CORR_METHODS = ['pearson', 'kendall', 'spearman']
MATRIX_PALETTE = 'RdYlBu'
LEGEND_LAYOUT = dict(yanchor='top', y=0.99, xanchor='right', x=0.99)

RolloutAnalysis = Dict[str, np.ndarray]
"""
The result of interestingness analysis for one rollout.
The key is dimension name, array is in form (batch, ).
"""

RolloutsAnalyses = Dict[str, Dict[str, np.ndarray]]
"""
The result of interestingness analysis for a set of rollouts.
The first key is dimension name, then key is rollout id, then array is (batch, ). 
"""

class AnalysisDimensionBase(ABC):
    """
    The base class for interestingness analysis dimensions. Analyses extract useful information from the agent's history
    of interaction with its environment as provided by an interaction dataset. Analyses allow the identification of
    interestingness elements, i.e., situations that may be interesting to help explain an agent's behavior and aptitude
    in some task, both in terms of its capabilities and limitations. An analysis class is given interaction data
    (list of interaction datapoints) for one episode as input. For each step in the episode, it produces one or more
    values, ideally in the [0,1] interval, denoting the “strength” or “amount” measured by the analysis with respect to
    the interestingness dimension. In addition, for each step it can also provide a label that qualitatively
    characterizes that step within the episode, based on thresholds and other criteria.
    """

    def __init__(self, name: str, num_processes: int = 1):
        """
        Creates a new analysis.
        :param str name: the name of the interestingness dimension extracted by this analysis.
        :param int num_processes: the number of parallel processes to use for this analysis. A value of `-1` or `None`
        will use all available cpus.
        """
        self.name = name
        self.num_processes = os.cpu_count() if num_processes == -1 or num_processes is None else num_processes

    @abstractmethod
    def analyze(self, rollouts: Rollouts) -> Optional[RolloutsAnalyses]:
        """
        Analyzes an agent's history of interaction with its environment according to this interestingness dimension and
        returns values denoting the “strength” or “amount” associated with each timestep.
        :param Rollouts rollouts: the dataset to be analyzed, corresponding to interaction data for each
        rollout/episode.
        :rtype: RolloutsAnalyses or None
        :return: a dictionary of the form `{dimension: {rollout_id: values}}` containing the interestingness values
        for each rollout (each an array of shape (timesteps, )), organized for each extracted dimension. If the
        analysis could not be performed (e.g., the necessary data is not present in the datapoints), it returns `None`.
        """
        pass

    def _analyze_parallel(self, func: Callable[[Rollout], Optional[RolloutAnalysis]], rollouts: Rollouts) \
            -> Optional[RolloutsAnalyses]:

        # performs analysis for each rollout/episode in parallel
        rollouts_res: List[Optional[RolloutAnalysis]] = run_parallel(
            func, list(rollouts.values()), self.num_processes, use_tqdm=True)

        # organizes data by dimension and episode
        results: RolloutsAnalyses = {}
        for i, rollout_id in enumerate(rollouts.keys()):
            rollout_res = rollouts_res[i]
            if rollout_res is None:
                continue  # ignore, no analysis data

            # check analysis dimension and appends data
            for dimension, res in rollout_res.items():
                if dimension not in results:
                    results[dimension] = {}
                results[dimension][rollout_id] = res
        return None if len(results) == 0 else results


def get_interestingness_dataframe(analyses_results: RolloutsAnalyses, interaction_data: Rollouts) -> \
        pd.DataFrame:
    """
    Gets a pandas DataFrame containing the interestingness analyses results over all episodes.
    :param RolloutsAnalyses analyses_results: the interestingness analyses results to be converted to a dataframe.
    :param Rollouts interaction_data: the interaction data containing the timesteps information.
    :rtype: pd.DataFrame
    :return: a pandas DataFrame with columns Rollout ID, Timestep, Dimension1, Dimension2, ... containing the
    interestingness analyses results over all timesteps of all rollouts/episodes.
    """
    dfs = []
    for dim, analyses in analyses_results.items():
        rollout_dfs = []
        for rollout_id, analysis in analyses.items():
            df = pd.DataFrame({ROLLOUT_ID_COL: rollout_id,
                               TIMESTEP_COL: interaction_data[rollout_id].data.timesteps,
                               dim: analysis})
            rollout_dfs.append(df)
        dfs.append(pd.concat(rollout_dfs, axis=0))  # merge rows
    df = pd.concat(dfs, axis=1)  # merge columns
    df = df.loc[:, ~df.columns.duplicated()]
    df.reset_index(drop=True, inplace=True)
    return df


def print_stats(df: pd.DataFrame,
                output_dir: str,
                dimensions: List[str],
                img_format: str = 'pdf',
                rollout_plots: bool = True,
                num_processes: int = -1) -> Dict[str, Dict[str, str]]:
    """
    Creates plots and data files with different statistics about the given interestingness dataset.
    :param pd.DataFrame df: the dataset containing the interestingness data, generated by `get_interestingness_dataframe`.
    :param str output_dir: the path to the directory in which to save results.
    :param str img_format: the format of the generated image files.
    :param list[str] dimensions: the names of the interestingness dimensions to be plotted together.
    :param bool rollout_plots: whether to plot results individually for each rollout.
    :param int num_processes: the number of parallel processes to use for plotting.
    :rtype: dict[str, dict[str, str]]
    :return: a dictionary with all generated `plotly` figures in JSON format, organized by rollout and figure label.
    :return:
    """
    dummy_plotly()  # just to clear imports

    df = df.set_index(TIMESTEP_COL)  # so that time axis appears correctly

    dimensions = [dim for dim in dimensions if dim in df.columns]
    all_dims: List[str] = [dim for dim in df.columns if dim not in [TIMESTEP_COL, ROLLOUT_ID_COL]]
    rollout_ids = df[ROLLOUT_ID_COL].unique()

    figures = {rollout_id: {} for rollout_id in rollout_ids}
    figures[MEAN_ROLLOUT_ID] = {}

    logging.info('_____________________________________')
    logging.info('Plotting mean interestingness...')
    int_df = df[[ROLLOUT_ID_COL] + dimensions]

    fig = plot_timeseries(int_df, 'Mean Interestingness',
                          os.path.join(output_dir, f'mean-interestingness-time.{img_format}'),
                          x_label='Timesteps', y_label='Value', var_label='Dimension',
                          average=True, group_by=ROLLOUT_ID_COL, y_min=-1, y_max=1, smooth_factor=SMOOTH_FACTOR,
                          show_legend=True, legend=LEGEND_LAYOUT)
    figures[MEAN_ROLLOUT_ID][fig.layout.title.text] = fig.to_json()  # stores figure

    int_df = df[dimensions]
    int_df.describe(include='all').to_csv(os.path.join(output_dir, f'stats.csv'))
    plot_bar(int_df, 'Mean Interestingness',
             os.path.join(output_dir, f'mean-interestingness-bar.{img_format}'),
             x_label='Dimension', y_label='Value')
    fig = plot_radar(int_df, 'Mean Interestingness',
                     os.path.join(output_dir, f'mean-interestingness-radar.{img_format}'),
                     var_label='Dimension', value_label='Value', min_val=-1, max_val=1)
    figures[MEAN_ROLLOUT_ID]['Overall Mean Interestingness'] = fig.to_json()  # stores figure

    logging.info('_____________________________________')
    logging.info('Plotting interestingness dimensions individually...')
    for dim in tqdm.tqdm(all_dims):
        int_df = df[[ROLLOUT_ID_COL, dim]]
        fig = plot_timeseries(int_df, f'Mean {dim.title()}',
                              os.path.join(output_dir, f'mean-{dim.lower()}-time.{img_format}'),
                              x_label='Timesteps', y_label='Value', var_label='Dimension',
                              average=True, group_by=ROLLOUT_ID_COL, smooth_factor=SMOOTH_FACTOR, show_legend=False)
        figures[MEAN_ROLLOUT_ID][fig.layout.title.text] = fig.to_json()  # stores figure

    logging.info('_____________________________________')
    logging.info('Plotting mean interestingness for each action factor...')
    for dim in tqdm.tqdm(dimensions):
        action_dims = [d for d in all_dims if dim in d and d != dim]
        if len(action_dims) == 0:
            continue  # no action-factor interestingness

        # bar plot
        act_df = df[action_dims].copy()
        act_df.columns = [act_dim.replace(dim, '').strip(' -') for act_dim in act_df.columns]
        fig = plot_bar(act_df, f'Mean Action Factors\' {dim.title()}',
                       os.path.join(output_dir, f'mean-{dim.lower()}-action-factors-bar.{img_format}'),
                       x_label='Action Factor', y_label=dim.title())
        figures[MEAN_ROLLOUT_ID][fig.layout.title.text + ' Bar'] = fig.to_json()  # stores figure

        # timeseries
        act_df = df[[ROLLOUT_ID_COL] + action_dims]
        act_df.columns = [act_dim.replace(dim, '').strip(' -') for act_dim in act_df.columns].copy()
        fig = plot_timeseries(act_df, f'Mean Action Factors\' {dim.title()}',
                              os.path.join(output_dir, f'mean-{dim.lower()}-action-factors-time.{img_format}'),
                              x_label='Timesteps', y_label=dim.title(), var_label='Action Factor',
                              average=True, group_by=ROLLOUT_ID_COL, smooth_factor=SMOOTH_FACTOR,
                              show_legend=True, legend=LEGEND_LAYOUT)
        figures[MEAN_ROLLOUT_ID][fig.layout.title.text + ' Time'] = fig.to_json()  # stores figure

    logging.info('_____________________________________')
    logging.info('Plotting correlation matrices for the interestingness dimensions...')
    int_df: pd.DataFrame = df[dimensions]
    for corr_method in tqdm.tqdm(CORR_METHODS):
        corr_mat = int_df.corr(corr_method)
        fig = plot_matrix(corr_mat, f'Interestingness Correlation ({corr_method.title()})',
                          os.path.join(output_dir, f'correlation-{corr_method.lower()}.{img_format}'),
                          z_min=-1, z_max=1, symmetrical=True, palette=MATRIX_PALETTE, height=680,
                          var_label=f'{corr_method.title()} Correlation')
        figures[MEAN_ROLLOUT_ID][fig.layout.title.text] = fig.to_json()  # stores fig

    # plot results for each rollout
    if rollout_plots:
        logging.info('_____________________________________')
        logging.info('Plotting interestingness for each rollout...')

        args = [(rollout_id, df, all_dims, dimensions) for rollout_id in rollout_ids]
        rollouts_figs = run_parallel(_plot_rollout, args, processes=num_processes, use_tqdm=True)
        for rollout_figs in rollouts_figs:
            figures.update(rollout_figs)

    return figures


def _plot_rollout(rollout_id, df, all_dims, dimensions):
    dummy_plotly()  # just to clear imports

    int_df = df[df[ROLLOUT_ID_COL] == rollout_id][dimensions]
    figures = {rollout_id: {}}

    # all dimensions
    fig = plot_timeseries(int_df, f'Rollout {rollout_id} - Interestingness',
                          x_label='Timesteps', y_label='Value', var_label='Dimension',
                          show_legend=True, legend=LEGEND_LAYOUT)
    figures[rollout_id]['Interestingness'] = fig.to_json()  # stores figure

    # mean interestingness
    fig = plot_radar(int_df, 'Mean Interestingness',
                     var_label='Dimension', value_label='Value', min_val=-1, max_val=1)
    figures[rollout_id]['Overall Mean Interestingness'] = fig.to_json()  # stores figure

    # individual dimensions
    for dim in all_dims:
        dim_df = df[df[ROLLOUT_ID_COL] == rollout_id][dim]
        fig = plot_timeseries(dim_df, f'Rollout {rollout_id} - {dim}',
                              x_label='Timesteps', y_label='Value', var_label='Dimension', show_legend=False)
        figures[rollout_id][dim] = fig.to_json()  # stores figure

    # action factor dimensions
    for dim in dimensions:
        action_dims = [d for d in all_dims if dim in d and d != dim]
        if len(action_dims) == 0:
            continue  # no action-factor interestingness

        act_df = df[df[ROLLOUT_ID_COL] == rollout_id][action_dims].copy()
        act_df.columns = [act_dim.replace(dim, '').strip(' -') for act_dim in act_df.columns]
        fig = plot_bar(act_df, f'Rollout {rollout_id} - Action Factors\' {dim}',
                       x_label='Action Factor', y_label=dim)
        figures[rollout_id][f'Action Factors {dim} Bar'] = fig.to_json()  # stores figure

        fig = plot_timeseries(act_df, f'Rollout {rollout_id} - Action Factors\' {dim}',
                              x_label='Timesteps', y_label=dim, var_label='Action Factor',
                              smooth_factor=SMOOTH_FACTOR, show_legend=True, legend=LEGEND_LAYOUT)
        figures[rollout_id][f'Action Factors {dim} Time'] = fig.to_json()  # stores figure

    return figures
