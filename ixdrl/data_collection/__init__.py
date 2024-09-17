import collections
import gymnasium as gym
import itertools as it
import json
import logging
import numpy as np
import os
import pandas as pd
import plotly.express as px
import tqdm
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict, List, Any, Callable, Optional

from ixdrl import Rollouts, CategoricalDistribution, Rollout, MultiData, MultiCategoricalDistribution, \
    AtomicData
from ixdrl.util.gym import RepeatedDiscrete, is_spatial
from ixdrl.util.math import running_average
from ixdrl.util.plot import plot_timeseries, plot_bar, dummy_plotly, plot_histogram, plot_matrix

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

INTERACTION_DATA_FILE = 'interaction_data.pkl.gz'
INTERACTION_PLOTS_FILE = 'data_plots.pkl.gz'
HIST_PALETTE = 'Portland'
DISC_PALETTE_10 = px.colors.qualitative.T10
DISC_PALETTE_24 = px.colors.qualitative.Light24
MATRIX_PALETTE = 'Inferno'
MEAN_ROLLOUT_ID = 'Mean'
ROLLOUT_COL = 'Rollout'
SMOOTH_FACTOR = 0.9


class InteractionDataCollector(ABC):
    """
    Represents objects that collect interaction data given some RL agent trained in some environment.
    The idea is to run a trained agent in the environments and register all data that can be useful to analyze the
    agent’s behavior and its competency in the task. This represents the first step towards the extraction of
    interestingness elements.
    """

    @abstractmethod
    def collect_data(self, num_rollouts: int) -> Rollouts:
        """
        Generates/collects all interaction data from an agent by generating some number of rollouts.
        :param int num_rollouts: the number of rollouts/episodes for which to collect data.
        :rtype: Rollouts
        :return: the data collected during the agent's interaction with the environment.
        """
        pass


def print_stats(interaction_data: Rollouts, output_dir: str, img_format: str = 'pdf') -> \
        Dict[str, Dict[str, str]]:
    """
    Creates plots and data files with different statistics about the given interaction data.
    :param Rollouts interaction_data: the interaction data to be analyzed.
    :param str output_dir: the path to the directory in which to save results.
    :param str img_format: the format of the generated image files.
    :rtype: dict[str, dict[str, str]]
    :return: a dictionary with all generated `plotly` figures in JSON format, organized by rollout and figure label.
    """
    dummy_plotly()  # just to clear imports

    rollout_ids = list(interaction_data.keys())
    figures = {rollout_id: {} for rollout_id in rollout_ids}
    figures[MEAN_ROLLOUT_ID] = {}

    logging.info('_____________________________________')
    logging.info('Plotting time steps...')
    df = pd.DataFrame({'Timesteps': [len(rollout.data.timesteps) for rollout in interaction_data.values()]})
    df.describe(include='all').to_csv(os.path.join(output_dir, 'stats-time.csv'))
    logging.info(f'Total timesteps: {df.sum().values[0]}')
    fig = plot_histogram(df, 'Rollout Length Histogram',
                         os.path.join(output_dir, f'time-hist.{img_format}'),
                         x_label='Timesteps', y_label='Count', plot_mean=True,
                         show_legend=False, palette=HIST_PALETTE)
    figures[MEAN_ROLLOUT_ID]['Rollout Length Histogram'] = fig.to_json()  # stores figure

    logging.info('_____________________________________')
    logging.info('Plotting reward...')
    df = pd.concat([pd.DataFrame({rollout_id: rollout.data.reward[:, 0]}, index=rollout.data.timesteps)
                    for rollout_id, rollout in interaction_data.items()], axis=1)
    _plot_timeseries(df, rollout_ids, figures, output_dir, img_format, 'Reward')

    df = pd.concat([pd.DataFrame({rollout_id: np.cumsum(rollout.data.reward[:, 0])}, index=rollout.data.timesteps)
                    for rollout_id, rollout in interaction_data.items()], axis=1)
    _plot_timeseries(df, rollout_ids, figures, output_dir, img_format, 'Cumulative Reward')

    df = pd.DataFrame({'Cumulative Reward': [np.sum(rollout.data.reward[:, 0])
                                             for rollout in interaction_data.values()]})
    df.describe(include='all').to_csv(os.path.join(output_dir, 'stats-reward-sum.csv'))
    plot_histogram(df, 'Cumulative Reward Histogram',
                   os.path.join(output_dir, f'reward-sum-hist.{img_format}'),
                   x_label='Cumulative Reward', y_label='Count', plot_mean=True,
                   show_legend=False, palette=HIST_PALETTE)
    figures[MEAN_ROLLOUT_ID]['Cumulative Reward Histogram'] = fig.to_json()  # stores figure

    logging.info('_____________________________________')
    logging.info('Plotting value...')
    sample_rollout = next(iter(interaction_data.values()))
    if isinstance(sample_rollout.data.value, CategoricalDistribution):
        # get the value's weighted average for each rollout
        df = pd.concat([pd.DataFrame(
            {rollout_id: np.sum(rollout.data.value.probs * rollout.data.value.support, axis=-1)},
            index=rollout.data.timesteps)
            for rollout_id, rollout in interaction_data.items()], axis=1)
    else:
        df = pd.concat([pd.DataFrame({rollout_id: rollout.data.value[:, 0]}, index=rollout.data.timesteps)
                        for rollout_id, rollout in interaction_data.items()], axis=1)
    _plot_timeseries(df, rollout_ids, figures, output_dir, img_format, 'Value')

    logging.info('_____________________________________')
    logging.info('Plotting observation stats...')
    obs_space = sample_rollout.observation_space  # assumes all obs spaces the same..
    obs_labels = sample_rollout.observation_labels
    _plot_data_for_space(interaction_data, lambda rollout: rollout.data.observation, obs_space, obs_labels, rollout_ids,
                         figures, output_dir, img_format, 'Observation', 'Frequency')

    logging.info('_____________________________________')
    logging.info('Plotting action stats...')
    act_space = sample_rollout.action_space  # assumes all action spaces the same..
    act_labels = sample_rollout.action_labels
    _plot_data_for_space(interaction_data, lambda rollout: rollout.data.action, act_space, act_labels, rollout_ids,
                         figures, output_dir, img_format, 'Action', 'Frequency')

    logging.info('_____________________________________')
    logging.info('Plotting action probability...')
    if isinstance(sample_rollout.data.action_prob, MultiData):
        for i, dim in enumerate(act_labels):
            logging.info(f'Plotting action probability for dimension {dim}...')
            dfs = []
            for rollout_id, rollout in interaction_data.items():
                probs = rollout.data.action_prob.data[i][:, 0]
                dfs.append(pd.DataFrame({rollout_id: probs}, index=rollout.data.timesteps))
            df = pd.concat(dfs, axis=1)
            _plot_timeseries(df, rollout_ids, figures, output_dir, img_format, f'{dim} Action Probability')

    else:
        df = pd.concat([pd.DataFrame({rollout_id: rollout.data.action_prob[:, 0]}, index=rollout.data.timesteps)
                        for rollout_id, rollout in interaction_data.items()], axis=1)
        _plot_timeseries(df, rollout_ids, figures, output_dir, img_format, 'Action Probability')

    logging.info('_____________________________________')
    logging.info('Plotting action distribution...')
    action_dist = sample_rollout.data.action_dist
    if isinstance(action_dist, MultiCategoricalDistribution):
        if isinstance(act_space, RepeatedDiscrete):
            cat_labels = np.full(act_space.shape + (len(act_space.labels),), act_space.labels).tolist()
        else:
            # generate action name labels directly from distribution
            cat_labels = [[f'Action {j}' for j in range(dist.probs.shape[-1])] for dist in action_dist.dists]

        for i, dim in enumerate(act_labels):
            logging.info(f'Plotting action distribution for dimension {dim}...')
            dfs = []
            for rollout_id, rollout in interaction_data.items():
                probs = rollout.data.action_dist.dists[i].probs  # shape: (timesteps, num_actions)
                df = pd.DataFrame(probs, index=rollout.data.timesteps, columns=cat_labels[i])
                df[ROLLOUT_COL] = rollout.rollout_id  # set rollout id
                dfs.append(df)
            df = pd.concat(dfs, axis=0)
            _plot_bar(df, figures, output_dir, img_format, f'{dim} Action Distribution', 'Action', 'Probability')

    elif isinstance(action_dist, CategoricalDistribution):
        dfs = []
        for rollout_id, rollout in interaction_data.items():
            probs = rollout.data.action_dist.probs  # shape: (timesteps, num_actions)
            df = pd.DataFrame(probs, index=rollout.data.timesteps, columns=act_labels)
            df[ROLLOUT_COL] = rollout.rollout_id  # set rollout id
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        _plot_bar(df, figures, output_dir, img_format, 'Action Distribution', 'Action', 'Probability')

    logging.info('_____________________________________')
    logging.info('Plotting action values...')
    action_vals = sample_rollout.data.action_values
    if isinstance(action_vals, CategoricalDistribution):
        for i, dim in enumerate(act_labels):
            dfs = []
            for rollout_id, rollout in interaction_data.items():
                probs = rollout.data.action_values.probs[:, i]  # shape: (timesteps, n_vals)
                df = pd.DataFrame(probs, index=rollout.data.timesteps, columns=rollout.data.action_values.support[0, i])
                df[ROLLOUT_COL] = rollout.rollout_id  # set rollout id
                dfs.append(df)
            df = pd.concat(dfs, axis=0)
            _plot_bar(df, figures, output_dir, img_format, f'Action {dim} Value Distribution', 'Value', 'Probability')
    elif isinstance(action_vals, AtomicData):
        dfs = []
        for rollout_id, rollout in interaction_data.items():
            vals = rollout.data.action_values  # shape: (timesteps, num_actions)
            df = pd.DataFrame(vals, index=rollout.data.timesteps, columns=act_labels)
            df[ROLLOUT_COL] = rollout.rollout_id  # set rollout id
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        _plot_bar(df, figures, output_dir, img_format, 'Action Values', 'Action', 'Value')

    return figures


def _plot_data_for_space(interaction_data: Rollouts, data_func: Callable[[Rollout], np.ndarray], space: gym.Space,
                         labels: Optional[List[str]], rollout_ids: List[str], figures: Dict[str, Dict[str, str]],
                         output_dir: str, img_format: str, x_label: str, y_label: str):
    """
    Generates plots for some data (action or observation) according to the given space.
    """
    if isinstance(space, gym.spaces.Discrete):
        # if discrete, plot mean per episode
        dfs = []
        for rollout in interaction_data.values():
            # organize data by counts for the different data values/categories (one-hot encoding style)
            values = data_func(rollout).flatten()
            dfs.append(pd.get_dummies(values.astype(str)).sum(axis=0, skipna=True))

        df = pd.concat(dfs, axis=1).T
        if labels is None:
            labels = [str(i) for i in range(space.n)]  # assumes labels are the indices
        else:
            rename_dict = {}
            labels = list(labels)  # make copy
            for i, label in enumerate(labels):
                if isinstance(label, IntEnum):
                    rename_dict[str(label.value)] = label.name  # replace int value with label
                    labels[i] = label.name  # set only enum name in label
                else:
                    rename_dict[str(i)] = str(label)  # try to replace index value with label
            df.rename(columns=rename_dict, inplace=True)

        palette = DISC_PALETTE_10 if len(labels) <= 10 else DISC_PALETTE_24 if len(labels) <= 24 else HIST_PALETTE
        _plot_discrete_values(
            df, labels, rollout_ids, figures, x_label, y_label, output_dir, img_format, palette)

    elif isinstance(space, gym.spaces.MultiDiscrete):
        # MultiDiscrete, plot mean distribution per dim per episode
        if labels is None:
            labels = [f'Dim {i}' for i in range(len(space.nvec))]  # generates dummy labels if not provided
        for i, dim in enumerate(labels):
            logging.info(f'Plotting {x_label.lower()} for dimension {dim}...')
            _plot_data_for_space(interaction_data, lambda r: data_func(r)[:, i], gym.spaces.Discrete(space.nvec[i]),
                                 None, rollout_ids, figures, output_dir, img_format, dim, y_label)

    elif isinstance(space, RepeatedDiscrete):
        cat_labels = space.labels if space.labels is not None else np.arange(np.iinfo(space.dtype).max)
        spatial, idxs = is_spatial(space)
        if spatial:
            # if spatial, plot mean array of value counts per episode for each dim
            axes = list(np.array(idxs) + 1)  # data is batch mode, so swap such that shape: (dims*, batch, w, h)
            axes = axes[:-2] + [0] + axes[-2:]
            data = collections.OrderedDict({rollout_id: data_func(rollout).transpose(axes)
                                            for rollout_id, rollout in interaction_data.items()})
            if labels is None:
                labels = [f'Dim {i}' for i in range(len(idxs))]  # generates dummy labels if not provided
            else:
                labels = list(np.array(labels)[idxs])  # swap order of dimension labels

            if len(idxs) == 2:
                _plot_spatial_discrete(
                    data, cat_labels, figures, output_dir, img_format, x_label, labels[0], labels[1], z_min=0)
            else:
                for comb in it.product(*[np.arange(s) for s in np.array(space.shape)[idxs][:-2]]):
                    _data = collections.OrderedDict({rollout_id: d[comb] for rollout_id, d in data.items()})
                    name = ''.join(f'{labels[i]} {j}' for i, j in enumerate(comb))
                    logging.info(f'Plotting {x_label.lower()} for {name}...')
                    _plot_spatial_discrete(
                        _data, cat_labels, figures, output_dir, img_format, name, labels[-2], labels[-1], z_min=0)
        else:
            # plot mean distribution per dim per episode
            if labels is None:
                labels = [f'Dim {i}' for i in range(space.shape[0])]  # generates dummy labels if not provided
            for i, dim in enumerate(labels):
                _x_label = x_label + (f' {dim}' if len(labels) > 1 else '')
                logging.info(f'Plotting {_x_label.lower()} for dimension {dim}...')
                _data_func = (lambda r: data_func(r)[:, i]) if len(labels) > 1 else data_func
                _plot_data_for_space(interaction_data, _data_func,
                                     gym.spaces.Discrete(len(cat_labels)), cat_labels, rollout_ids, figures,
                                     output_dir, img_format, _x_label, y_label)

    elif isinstance(space, gym.spaces.Box):
        spatial, idxs = is_spatial(space)
        if spatial:
            # if spatial, plot mean array per timestep for each dim
            axes = list(np.array(idxs) + 1)  # data is batch mode, so swap such that shape: (dims*, batch, w, h)
            axes = axes[:-2] + [0] + axes[-2:]
            data = collections.OrderedDict({rollout_id: data_func(rollout).transpose(axes)
                                            for rollout_id, rollout in interaction_data.items()})
            if labels is None:
                labels = [f'Dim {i}' for i in range(len(idxs))]  # generates dummy labels if not provided
            else:
                labels = list(np.array(labels)[idxs])  # swap order of dimension labels

            dtype = next(iter(data.values())).dtype
            z_min = np.min(space.low if space.dtype == dtype else None)
            z_max = np.max(space.high if space.dtype == dtype else None)
            if len(idxs) == 2:
                _plot_spatial_continuous(
                    data, figures, output_dir, img_format, x_label, labels[0], labels[1], z_min, z_max)
            else:
                for comb in it.product(*[np.arange(s) for s in np.array(space.shape)[idxs][:-2]]):
                    _data = collections.OrderedDict({rollout_id: obs[comb] for rollout_id, obs in data.items()})
                    name = ''.join(f'{labels[i]} {j}' for i, j in enumerate(comb))
                    logging.info(f'Plotting {x_label.lower()} for {name}...')
                    _plot_spatial_continuous(
                        _data, figures, output_dir, img_format, name, labels[-2], labels[-1], z_min, z_max)
        else:
            if labels is None:
                labels = [f'Dim {i}' for i in range(space.shape[0])]  # generates dummy labels if not provided

            # otherwise plot mean values per timestep
            df = pd.DataFrame(collections.OrderedDict({
                label: np.concatenate([data_func(rollout).reshape(data_func(rollout).shape[0], -1)[:, i]
                                       for rollout in interaction_data.values()])
                for i, label in enumerate(labels)
            }))
            # add rollout ID column
            df[ROLLOUT_COL] = np.concatenate([[rollout_id] * data_func(rollout).shape[0]
                                              for rollout_id, rollout in interaction_data.items()])
            _plot_continuous_values(df, rollout_ids, figures, output_dir, img_format, f'{x_label} Dimension')

    elif isinstance(space, gym.spaces.Tuple):
        # if tuple, plot for each sub-space
        if labels is None:
            labels = [f'Subspace {i}' for i in range(len(space.spaces))]  # generates dummy labels if not provided
        for i, subspace in enumerate(labels):
            subspace = labels[i]
            _plot_data_for_space(interaction_data, lambda r: data_func(r)[:, i], space.spaces[i], None,
                                 rollout_ids, figures, output_dir, img_format, subspace, y_label)

    else:
        logging.warning(f'Cannot handle {x_label.lower()} space: {space}')


def _plot_spatial_discrete(data: Dict[str, np.ndarray], labels: List[str], figures: Dict[str, Dict[str, str]],
                           output_dir: str, img_format: str, name: str, x_label: str, y_label: str,
                           z_min: Optional[float] = None, z_max: Optional[float] = None):
    """
    Plots a discrete spatial quantity as a heatmap, one for each discrete value/category.
    Plots the average value over all timesteps of all episodes to file.
    Plots the average value over all timesteps for each episode but does not save them to file.
    Adds all plots to the dictionary of figures.
    :param data: dictionary with each episode data in batch format: (timesteps, width, height)
    """
    # compute discrete values to labels mapping
    value_label_map = {}
    for i, label in enumerate(labels):
        if isinstance(label, IntEnum):
            value_label_map[label.value] = label.name  # replace int value with label
        elif 'int' in str(type(label)) or 'float' in str(type(label)):
            value_label_map[label] = str(label)  # replace int value with string
        elif isinstance(label, str) and label.replace('.', '', 1).isdigit():
            value_label_map[float(label)] = label  # replace numeric value with string
        else:
            value_label_map[i] = str(label)  # try to replace index value with label

    for value, label in value_label_map.items():
        # gets mean for each rollout for this category value
        _data = collections.OrderedDict({rollout_id: d == value for rollout_id, d in data.items()})
        _name = f'{name}={label}'
        logging.info(f'Plotting {_name}...')
        _plot_spatial_continuous(_data, figures, output_dir, img_format, _name, x_label, y_label, z_min, z_max)


def _plot_spatial_continuous(data: Dict[str, np.ndarray], figures: Dict[str, Dict[str, str]],
                             output_dir: str, img_format: str, name: str, x_label: str, y_label: str,
                             z_min: Optional[float] = None, z_max: Optional[float] = None):
    """
    Plots a continuous spatial quantity as a heatmap.
    Plots the average value over all timesteps of all episodes to file.
    Plots the average value over all timesteps for each episode but does not save them to file.
    Adds all plots to the dictionary of figures.
    :param data: dictionary with each episode data in batch format: (timesteps, width, height)
    """
    file_suffix = name.lower().replace(' ', '-').replace('=', '-')

    # running average since some datasets can be huge..
    mean = running_average(list(data.values()), axis=0)

    height = 600
    width = int(height * (mean.shape[1] / mean.shape[0] + .1))  # adjust width such that legend is not far

    title = f'Mean {name} per Timestep'
    fig = plot_matrix(mean, title,
                      os.path.join(output_dir, f'mean-{file_suffix}.{img_format}'),
                      show_values=False, z_min=z_min, z_max=z_max, width=width, height=height,
                      x_label=x_label, y_label=y_label, palette=MATRIX_PALETTE)
    figures[MEAN_ROLLOUT_ID][title] = fig.to_json()  # stores figure

    # generate individual rollout plots, do not save to file
    for rollout_id, d in tqdm.tqdm(data.items()):
        fig = plot_matrix(np.mean(d, axis=0), f'Rollout {rollout_id} - {title}',
                          show_values=False, z_min=z_min, z_max=z_max, width=width, height=height,
                          x_label=x_label, y_label=y_label, palette=MATRIX_PALETTE)
        figures[rollout_id][name] = fig.to_json()


def _plot_bar(df: pd.DataFrame, figures: Dict[str, Dict[str, str]],
              output_dir: str, img_format: str, title: str, x_label: str, y_label: str):
    """
    Plots the mean of a discrete variable that varies over time grouped by rollout/episode (bar plots).
    Plots the average value over all timesteps and episodes to file. Plots individual plots for each episode but does
    not save them to file. Adds all plots to the dictionary of figures.
    :param df: dataframe with the values for each timestep (rows) for each category (columns, including rollout id).
    """
    _title = f'Mean {title}'
    file_suffix = title.lower().replace(' ', '-')
    df.describe(include='all').to_csv(os.path.join(output_dir, f'stats-{file_suffix}.csv'))

    labels = [col for col in df.columns if col != ROLLOUT_COL]
    palette = DISC_PALETTE_10 if len(labels) <= 10 else DISC_PALETTE_24 if len(labels) <= 24 else HIST_PALETTE
    fig = plot_bar(df[labels], _title, os.path.join(output_dir, f'mean-{file_suffix}.{img_format}'),
                   x_label=x_label, y_label=y_label, palette=palette)
    figures[MEAN_ROLLOUT_ID][_title] = fig.to_json()  # stores figure

    # generate individual rollout plots, do not save to file
    for rollout_id, _df in tqdm.tqdm(df.groupby(ROLLOUT_COL), total=len(df[ROLLOUT_COL].unique())):
        fig = plot_bar(_df[labels], f'Rollout {rollout_id} - {title}',
                       x_label=x_label, y_label=y_label, palette=palette)
        figures[rollout_id][title] = fig.to_json()


def _plot_timeseries(df: pd.DataFrame, rollout_ids: List[str], figures: Dict[str, Dict[str, str]],
                     output_dir: str, img_format: str, y_label: str, x_label: str = 'Timestep'):
    """
    Plots a quantity that varies over time for each rollout/episode (timeseries plots).
    Plots the average value over all episodes to file. Plots individual progression for each episode but does not save
    them to file. Adds all plots to the dictionary of figures.
    :param df: dataframe with the values for each timestep (rows) for each episode (columns).
    """
    file_suffix = y_label.lower().replace(' ', '-')
    df.describe(include='all').to_csv(os.path.join(output_dir, f'stats-{file_suffix}.csv'))

    title = f'Mean {y_label}'
    fig = plot_timeseries(df, title,
                          os.path.join(output_dir, f'mean-{file_suffix}.{img_format}'),
                          average=True, x_label=x_label, y_label=y_label, var_label='Episode',
                          smooth_factor=SMOOTH_FACTOR, show_legend=False)
    figures[MEAN_ROLLOUT_ID][title] = fig.to_json()  # stores figure

    # generate individual rollout plots, do not save to file
    for rollout_id in tqdm.tqdm(rollout_ids):
        fig = plot_timeseries(df[rollout_id].to_frame(), f'Rollout {rollout_id} - {y_label}',
                              x_label=x_label, y_label=y_label, var_label='Episode',
                              smooth_factor=SMOOTH_FACTOR, show_legend=False)
        figures[rollout_id][y_label] = fig.to_json()


def _plot_discrete_values(df: pd.DataFrame, labels: List[str], rollout_ids: List[str],
                          figures: Dict[str, Dict[str, str]], x_label: str, y_label: str,
                          output_dir: str, img_format: str, palette: str):
    """
    Plots a categorical quantity as bar plot, each bar representing the mean value of a category.
    Plots the average value for each category over all episodes to file. Plots values for each category for each episode
    but does not save them to file. Adds all plots to the dictionary of figures.
    :param df: dataframe with the values for each episode (rows) for each category index (columns).
    """
    df.columns = [str(col) for col in df.columns]  # sanity check
    labels = [str(label) for label in labels]  # sanity check
    for lbl in labels:
        if lbl not in df.columns:
            df[lbl] = 0  # insert new column for category label with 0 count

    unknown_labels = set(df.columns) - set(labels)
    if len(unknown_labels) > 1 or (len(unknown_labels) == 1 and '0' not in unknown_labels):
        logging.warning(f'Ignoring labels for {x_label} {y_label}: {unknown_labels}')

    df = df[labels]  # only interested in the given list of labels
    file_suffix = f'{x_label}-{y_label}'.lower().replace(' ', '-')
    df.describe(include='all').to_csv(os.path.join(output_dir, f'stats-{file_suffix}.csv'))

    title = f'Mean {x_label} {y_label} per Episode'
    fig = plot_bar(df, title,
                   os.path.join(output_dir, f'mean-{file_suffix}.{img_format}'),
                   x_label=x_label, y_label=y_label, palette=palette)
    figures[MEAN_ROLLOUT_ID][title] = fig.to_json()  # stores figure

    # generate individual rollout plots
    for i, rollout_id in tqdm.tqdm(enumerate(rollout_ids), total=len(rollout_ids)):
        fig = plot_bar(df.iloc[i].to_frame().T, f'Rollout {rollout_id} - {x_label} {y_label}',
                       x_label=x_label, y_label=y_label, palette=palette)
        figures[rollout_id][f'{x_label} {y_label}'] = fig.to_json()


def _plot_continuous_values(df: pd.DataFrame, rollout_ids: List[str],
                            figures: Dict[str, Dict[str, str]], output_dir: str, img_format: str,
                            x_label: str, y_label: str = None):
    """
    Plots a vector of continuous values as bar plot, each bar representing the mean value of a dimension per timestep.
    Plots the average value for each dimension over all timesteps of all episodes to file.
    Plots the average value for each dimension over all timestep for each episode but does not save them to file.
    Adds all plots to the dictionary of figures.
    :param df: dataframe with the values for each timestep (rows) for each dimension (columns).
    Extra column for episode id to allow segmenting data for each episode.
    """
    file_suffix = (f'{x_label}-{y_label}' if y_label is not None else x_label).lower().replace(' ', '-')
    _df = df.loc[:, df.columns != ROLLOUT_COL]
    _df.describe(include='all').to_csv(os.path.join(output_dir, f'stats-{file_suffix}.csv'))

    title = f'Mean {x_label}{(" " + y_label) if y_label is not None else ""} per Step'
    fig = plot_bar(_df, title,
                   os.path.join(output_dir, f'mean-{file_suffix}.{img_format}'),
                   x_label=x_label, y_label=y_label)
    figures[MEAN_ROLLOUT_ID][title] = fig.to_json()  # stores figure

    # generate individual rollout plots
    title = x_label + (" " + y_label) if y_label is not None else ""
    for rollout_id in tqdm.tqdm(rollout_ids):
        _df = df[df[ROLLOUT_COL] == rollout_id].loc[:, df.columns != ROLLOUT_COL]
        fig = plot_bar(_df, f'Rollout {rollout_id} - Mean {title} Step', x_label=x_label, y_label=y_label)
        figures[rollout_id][title] = fig.to_json()


def save_metadata(rollouts: Rollouts, file_path: str):
    """
    Saves metadata about a set of rollouts to a JSON file.
    :param Rollouts rollouts: the rollouts whose metadata we want to save.
    :param str file_path: the path to the JSON file in which to save the metadata.
    :return:
    """
    # collect metadata common to all rollouts
    rollout = next(iter(rollouts.values()))
    metadata: Dict[str, Any] = collections.OrderedDict(dict(
        env_id=rollout.env_id,
        action_space=str(rollout.action_space),
        action_labels=rollout.action_labels,
        observation_space=str(rollout.observation_space),
        observation_labels=rollout.observation_labels,
        rwd_range=rollout.rwd_range,
        discount=rollout.discount))

    # collect metadata specific for each rollout
    rollout_data = metadata['rollouts'] = []
    for rollout_id, rollout in rollouts.items():
        rollout_data.append(collections.OrderedDict(dict(
            id=rollout_id,
            seed=rollout.seed,
            video_file=rollout.video_file,
        )))

    # save to json file
    with open(file_path, 'w') as fp:
        json.dump(metadata, fp, indent=4)
