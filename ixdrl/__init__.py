import gymnasium as gym
import tqdm
import numpy as np
import itertools as it
from typing import List, Dict, Tuple, Optional, Callable, Any, OrderedDict
from ixdrl.types import CategoricalDistribution, MultiCategoricalDistribution, NormalDistribution, \
    UniformDistribution, AtomicData, Distribution, MultiData, Data
from ixdrl.util.gym import RepeatedDiscrete

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class InteractionData(object):
    """
    An object that comprises information about the agent's interaction with its environment for a number of discrete
    timesteps (i.e., a rollout).
    """

    def __init__(self,
                 observation: AtomicData,
                 action: AtomicData,
                 reward: AtomicData,
                 value: Optional[Data] = None,
                 action_values: Optional[Data] = None,
                 action_dist: Optional[Distribution] = None,
                 action_prob: Optional[Data] = None,
                 pred_obs: Optional[Data] = None,
                 pred_rwd: Optional[Data] = None,
                 pred_horizon: Optional[int] = None,
                 timesteps: Optional[List[int]] = None,
                 user_data: Optional[Dict[str, Any]] = None):
        """
        Creates a new interaction data object.
        :param observation: the agent's observation at each step, shape (*, obs_space_shape).
        :param action: the action executed by the agent given each observation, shape (*, act_space_shape).
        :param reward: the reward received by the agent at the same time as the observation was taken, shape (*, 1).
        :param value: the value associated with each observation, can represent a distribution.
        :param action_values: the value associated with each action given each observation, can represent a distribution.
        :param action_dist: the probability distribution over actions given each observation.
        :param action_prob: the probability associated with each executed action and dimension, shape (*, act_dims).
        :param pred_obs: what the agent predicted the world would look like after executing each action given each
        observation. This is typically only available for model-based methods and can represent a distribution.
        :param pred_rwd: what the agent predicted the reward would be after executing the action given the
        observation. This is typically only available for model-based methods and can represent a distribution.
        :param pred_horizon: the horizon at which each predicted observation and reward were made.
        :param timesteps: the discrete timesteps corresponding to each sample in the data.
        :param user_data: some user-defined data for this datapoint.
        """
        # metadata
        self.timesteps: Optional[List[int]] = timesteps
        self.user_data: Optional[Dict[str, Any]] = user_data

        # MDP data
        self.observation: AtomicData = observation
        self.action: AtomicData = action
        self.reward: AtomicData = reward

        # RL data
        self.value: Optional[Data] = value
        self.action_values: Optional[Data] = action_values
        self.action_dist: Optional[Distribution] = action_dist
        self.action_prob: Optional[Data] = action_prob

        self.pred_obs: Optional[Data] = pred_obs
        self.pred_rwd: Optional[Data] = pred_rwd
        self.pred_horizon: Optional[int] = pred_horizon

        # check consistency
        if not self.check_consistency():
            raise ValueError('The batch size is not consistent across the provided interaction data items.')

    def check_consistency(self) -> bool:
        """
        Checks whether the items in this data object correspond to batch arrays and that they have a consistent size.
        :rtype: bool
        :return: `True` if all items contain batch data of consistent size, `False` otherwise.
        """
        # get batch size from observation item
        if self.observation is None or len(self.observation.shape) < 2:
            return False
        batch_size = self.observation.shape[0]

        def _is_consistent(data: Data) -> bool:
            if data is None:
                return True
            if isinstance(data, AtomicData):
                return len(data.shape) > 1 and data.shape[0] == batch_size
            if isinstance(data, MultiData):
                return all(_is_consistent(d) for d in data.data)
            if isinstance(data, CategoricalDistribution):
                return _is_consistent(data.probs) and _is_consistent(data.support)
            if isinstance(data, MultiCategoricalDistribution):
                return all(_is_consistent(d) for d in data.dists)
            if isinstance(data, NormalDistribution):
                return _is_consistent(data.mean) and _is_consistent(data.std)
            if isinstance(data, UniformDistribution):
                return _is_consistent(data.lower) and _is_consistent(data.upper)
            raise ValueError(f'Unknown data type: {data}.')

        # check if all items are consistent
        return (_is_consistent(self.action) and
                _is_consistent(self.reward) and
                _is_consistent(self.value) and
                _is_consistent(self.action_values) and
                _is_consistent(self.action_dist) and
                _is_consistent(self.action_prob) and
                _is_consistent(self.pred_obs) and
                _is_consistent(self.pred_rwd))


class Rollout(object):
    """
    Contains the data captured from an RL agent's rollout (e.g., an episode), including the collected interaction data.
    """

    def __init__(self, rollout_id: str,
                 data: List[InteractionData],
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 rwd_range: Tuple[float, float],
                 discount: float = None,
                 action_labels: List[str] = None,
                 observation_labels: List[str] = None,
                 video_file: str = None,
                 env_id: str = None,
                 seed: int = 0):
        """
        Creates a rollout data object.
        :param str rollout_id: the identifier for the rollout.
        :param list[InteractionData] data: the interaction data, one datapoint for each rollout timestep.
        :param gym.Space observation_space: the observation space of the environment for which data was collected.
        :param gym.Space action_space: the action space of the environment for which data was collected.
        :param (float, float) rwd_range: the reward value range.
        :param float discount: the discount factor of the MDP.
        :param list[str] action_labels: the names of the actions / dimensions.
        :param list[str] observation_labels: the names of the observation features.
        :param dir video_file: the path to the video file showing the agent's performance during this rollout.
        :param str env_id: the id of the environment for which data was collected.
        :param int seed: the random seed used to generate the rollout and corresponding interaction data.
        """
        self.rollout_id: str = rollout_id
        self.data: InteractionData = merge_datapoints(data, use_tqdm=True)
        self.observation_space: gym.Space = observation_space
        self.action_space: gym.Space = action_space
        self.rwd_range: Tuple[float, float] = rwd_range
        self.discount: float = discount
        self.action_labels: List[str] = action_labels
        self.observation_labels: List[str] = observation_labels
        self.video_file: str = video_file
        self.env_id: str = env_id
        self.seed: int = seed


def merge_datapoints(datapoints: List[InteractionData], use_tqdm: bool = True) -> InteractionData:
    """
    Merges the given interaction data points into a single dataset where items are in batch format.
    :param datapoints: the datapoints whose item is to be vectorized.
    :param use_tqdm: whether to use the tqdm module when iterating over datapoints.
    :rtype: InteractionData
    :return: an interaction data object containing the vectorized/batch versions of each data item.
    """
    if len(datapoints) == 1:
        return datapoints[0]

    def _merge_data(attr_func: Callable[[InteractionData], Data]):
        merged_data = {}

        def _add_data(dtype, d, **kwargs):
            # check type and initialize
            if 'type' not in merged_data:
                merged_data['type'] = dtype
                for k, v in kwargs.items():
                    merged_data[k] = []
            elif merged_data['type'] is not dtype:
                # data type mismatch
                raise ValueError(f'Expected data of type "{merged_data["type"]}", but {d} given.')

            # verify shapes and add to data
            for k, v in kwargs.items():
                if len(merged_data[k]) > 0:
                    if (len(v.shape) != len(merged_data[k][-1].shape) or
                            v.shape[1:] != merged_data[k][-1].shape[1:]):
                        raise ValueError(f'Shape mismatch while merging: {v.shape}, {merged_data[k][-1].shape}')
                merged_data[k].append(v)  # add to data

        for datapoint in (tqdm.tqdm(datapoints) if use_tqdm else datapoints):
            # check attribute data
            data = attr_func(datapoint)
            if data is None:
                return None

            # check data type, merge accordingly
            if isinstance(data, AtomicData):
                _add_data(AtomicData, data, data=data)
            elif isinstance(data, MultiData):
                _add_data(MultiData, data, **{f'{i}': data.data[i] for i in range(len(data.data))})
            elif isinstance(data, CategoricalDistribution):
                _add_data(CategoricalDistribution, data, probs=data.probs, support=data.support)
            elif isinstance(data, MultiCategoricalDistribution):
                kwargs = {f'{i}_probs': data.dists[i].probs for i in range(len(data.dists))}
                kwargs.update({f'{i}_supp': data.dists[i].support for i in range(len(data.dists))})
                _add_data(MultiCategoricalDistribution, data, **kwargs)
            elif isinstance(data, NormalDistribution):
                _add_data(NormalDistribution, data, mean=data.mean, std=data.std)
            elif isinstance(data, UniformDistribution):
                _add_data(UniformDistribution, data, lower=data.lower, upper=data.upper)
            elif isinstance(data, list):
                _add_data(list, data, **{f'{i}': data[i] for i in range(len(data))})
            else:
                raise ValueError(f'Unknown data type: {data}.')

        # creates vectorized versions of the data
        dtype = merged_data['type']
        if dtype is AtomicData:
            return np.concatenate(merged_data['data'], axis=0)
        if dtype is MultiData:
            return MultiData([np.concatenate(merged_data[f'{i}'], axis=0)
                              for i in range(len(merged_data.values()) - 1)])
        if dtype is CategoricalDistribution:
            return CategoricalDistribution(np.concatenate(merged_data['probs'], axis=0),
                                           np.concatenate(merged_data['support'], axis=0))
        if dtype is MultiCategoricalDistribution:
            return MultiCategoricalDistribution([CategoricalDistribution(
                np.concatenate(merged_data[f'{i}_probs'], axis=0), np.concatenate(merged_data[f'{i}_supp'], axis=0))
                for i in range(int((len(merged_data.values()) - 1) / 2))])
        if dtype is NormalDistribution:
            return NormalDistribution(np.concatenate(merged_data['mean'], axis=0),
                                      np.concatenate(merged_data['std'], axis=0))
        if dtype is UniformDistribution:
            return UniformDistribution(np.concatenate(merged_data['lower'], axis=0),
                                       np.concatenate(merged_data['upper'], axis=0))
        if dtype is list:
            return [np.concatenate(merged_data[f'{i}'], axis=0) for i in range(len(merged_data.values()) - 1)]

    # creates new interaction dataset from merged data for each attribute
    user_data = {k: _merge_data(lambda dp: dp.user_data[k])
                 for k in datapoints[0].user_data.keys()} if datapoints[0].user_data is not None else None
    return InteractionData(
        observation=_merge_data(lambda dp: dp.observation),
        action=_merge_data(lambda dp: dp.action),
        reward=_merge_data(lambda dp: dp.reward),
        value=_merge_data(lambda dp: dp.value),
        action_values=_merge_data(lambda dp: dp.action_values),
        action_dist=_merge_data(lambda dp: dp.action_dist),
        action_prob=_merge_data(lambda dp: dp.action_prob),
        pred_obs=_merge_data(lambda dp: dp.pred_obs),
        pred_rwd=_merge_data(lambda dp: dp.pred_rwd),
        timesteps=list(it.chain(*(dp.timesteps for dp in datapoints))),
        user_data=user_data
    )


def get_num_action_dimensions(action_space: gym.Space) -> int:
    """
    Gets the number of action factors/dimensions from the given action space.
    :param spaces.Space action_space: the action space specification.
    :rtype: int
    :return: the number of action dimensions consistent with the given action space.
    """
    if isinstance(action_space, gym.spaces.Discrete):
        return 1
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        return action_space.shape[0]
    if isinstance(action_space, gym.spaces.Box):
        return np.prod(action_space.shape, dtype=int).item()
    if isinstance(action_space, gym.spaces.Tuple):
        return len(action_space.spaces)
    if isinstance(action_space, RepeatedDiscrete):
        return int(np.prod(action_space.shape))

    raise NotImplementedError(f'Cannot determine number of actions from action space: {action_space}')


"""
Comprises a set of rollouts collected for some agent and task.
"""
Rollouts = OrderedDict[str, Rollout]
