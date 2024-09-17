import gymnasium as gym
import gymnasium.spaces as spaces
import itertools as it
import numpy as np
from typing import Union, List, Tuple, Optional

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class RepeatedDiscrete(gym.spaces.Space):
    """
    Represents a (repeated) discrete space. This can be useful for spatial maps with a fixed number of discrete values.
    """

    def __init__(self,
                 shape: Tuple[int, ...],
                 labels: Optional[List[Union[int, float, str]]] = None,
                 dtype=np.uint,
                 seed=None):
        """
        Creates a new repeated discrete space.
        :param (int,) shape: the shape of the space.
        :param list[int or float or str] labels: a list containing the possible discrete values (categories' labels).
        :param dtype: the data type.
        :param seed: the random seed.
        """
        super().__init__(shape, dtype, seed)
        self.labels = labels

    def sample(self):
        # generate random vector of possible indices
        high = len(self.labels) if self.labels is not None else np.iinfo(self.dtype).max
        return self.np_random.randint(0, high, self.shape, dtype=self.dtype)

    def contains(self, x):
        # checks shape, type, and whether all indices are valid (up to num. labels)
        high = len(self.labels) if self.labels is not None else np.iinfo(self.dtype).max
        return (x.shape == self.shape and
                x.dtype == self.dtype and
                (self.labels is None or all(val < high for val in np.unique(x))))

    def __repr__(self):
        return f'RepeatedDiscrete({self.shape}, {self.labels}, {self.dtype})'


def get_action_labels(env: Union[str, gym.Env, gym.envs.registration.EnvSpec]) -> List[Union[str, List[str]]]:
    """
    Gets names for the actions (for discrete action spaces) or action dimensions (for continuous action spaces)
    associated with the given environment.
    :param env: the environment name, spec, or gym identifier from which to get the action names.
    :rtype: list[str]
    :return: a list containing the labels for the actions / action dimensions. If no action labels are known for the
    given environment, a generic label is generated for each action / dimension.
    """
    if isinstance(env, gym.Env):
        # first check database
        if env.unwrapped.spec.id in ACTION_LABELS:
            return ACTION_LABELS[env.unwrapped.spec.id]

        # check ATARI, envs have this function which returns the action names
        if hasattr(env.unwrapped, 'get_action_meanings'):
            return [act_lbl.title() for act_lbl in env.unwrapped.get_action_meanings()]

        # check PyBullet, get action names from joints
        if hasattr(env.unwrapped, 'ordered_joints') and isinstance(env.unwrapped.ordered_joints, list):
            return [j.joint_name for j in env.unwrapped.ordered_joints]

        # check MuJoCo, get action names from actuators or joints
        if (hasattr(env.unwrapped, 'model') and hasattr(env.unwrapped.model, 'actuator_names') and
                isinstance(env.unwrapped.model.actuator_names, tuple)):
            if len(env.unwrapped.model.actuator_names) > 0:
                return [joint.replace('_', ' ') for joint in env.unwrapped.model.actuator_names]
            else:
                # resort to joints, ignore root x, z, y
                return [joint.replace('_', ' ') for joint in env.unwrapped.model.joint_names[3:]]

        # otherwise create generic action labels from space
        return _generate_dummy_labels(env.unwrapped.action_space, 'Action')

    if isinstance(env, gym.envs.registration.EnvSpec):
        return get_action_labels(env.id)
    if isinstance(env, str):
        return get_action_labels(gym.make(env))  # if id string, create env
    else:
        raise ValueError(f'Unknown environment type: {env}')


def get_observation_labels(env: Union[str, gym.Env, gym.envs.registration.EnvSpec]) -> List[Union[str, List[str]]]:
    """
    Gets names for the observation features associated with the given environment.
    :param env: the environment name, spec, or gym identifier from which to get the observation feature labels.
    :rtype: list[str]
    :return: a list containing the labels for the observation features. If no observation labels are known for the
    given environment, a generic label is generated for each feature.
    """
    if isinstance(env, gym.Env):
        # first check database
        if env.unwrapped.spec.id in OBSERVATION_LABELS:
            return OBSERVATION_LABELS[env.unwrapped.spec.id]

        # check PyBullet Walker type, get obs names
        if hasattr(env.unwrapped, 'robot') and hasattr(env.unwrapped.robot, 'walk_target_x'):
            robot = env.unwrapped.robot
            # see pybullet_envs.robot_locomotors.WalkerBase.calc_state
            more = ['Relative z', 'Sin(angle to target)', 'Cos(angle to target)',
                    'Velocity x', 'Velocity y', 'Velocity z',
                    'Body Roll', 'Body Pitch']
            j = list(it.chain(*[[f'{j.joint_name} Angle', f'{j.joint_name} Angular Velocity']
                                for j in robot.ordered_joints]))
            feet = [f'{f} Contact' for f in robot.foot_list]
            return more + j + feet

        # check MuJoCo, get names from joints (usually root x pos is ignored)
        if (hasattr(env.unwrapped, 'model') and hasattr(env.unwrapped.model, 'joint_names') and
                isinstance(env.unwrapped.model.joint_names, tuple)):
            return [f'{joint.replace("_", " ")} Position' for joint in env.unwrapped.model.joint_names[1:]] + \
                [f'{joint.replace("_", " ")} Velocity' for joint in env.unwrapped.model.joint_names]

        obs_shape = env.observation_space.shape
        # check ATARI via rllib
        if len(obs_shape) == 3 and hasattr(env, 'k') and obs_shape[2] == env.k:
            # see ray.rllib.env.wrappers.atari_wrappers.FrameStack
            return ['Pixel x', 'Pixel y', 'Frame stack']

        # check if obs space *seems* to be an image representation
        if isinstance(env.observation_space, gym.spaces.Box) and len(obs_shape) >= 2:
            highs = np.unique(env.observation_space.high)
            lows = np.unique(env.observation_space.low)
            if len(highs) == 1 and highs[0] == 255 and len(lows) == 1 and lows[0] == 0:
                labels = ['Pixel x', 'Pixel y']
                if len(obs_shape) == 3 and 3 <= obs_shape[2] <= 4:
                    labels.append('RGB channel')
                else:
                    for i in range(len(obs_shape) - 2):
                        labels.append(f'Dim {i}')
                return labels

        # otherwise create generic obs labels from space
        return _generate_dummy_labels(env.unwrapped.action_space, 'Obs')

    if isinstance(env, gym.envs.registration.EnvSpec):
        return get_observation_labels(env.id)
    if isinstance(env, str):
        return get_observation_labels(gym.make(env))  # if id string, create env
    else:
        raise ValueError(f'Unknown environment type: {env}')


def _generate_dummy_labels(space: gym.Space, prefix: str) -> List[Union[str, List[str]]]:
    if isinstance(space, spaces.Discrete):
        return [f'{prefix} {i}' for i in range(space.n)]
    if isinstance(space, spaces.MultiDiscrete):
        return [[f'{prefix} ({i},{j})' for j in range(size)] for i, size in enumerate(space.nvec)]
    if isinstance(space, spaces.Box):
        return [f'{prefix} Dim {i}' for i in range(len(space.shape) if len(space.shape) > 1 else space.shape[0])]
    return ['Unknown']


"""
Stores the labels of actions / action dimensions (factors) for known environments.  
"""
ACTION_LABELS = {
    # gym.envs.classic_control.cartpole.CartPoleEnv
    'CartPole-v0': ['Left', 'Right'],

    # gym.envs.classic_control.acrobot.AcrobotEnv
    'Acrobot-v1': ['+1 Torque', '0 Torque', '-1 Torque'],

    # https://github.com/openai/gym/wiki/Pendulum-v0
    'Pendulum-v1': ['Joint Effort'],

    # https://github.com/openai/gym/wiki/MountainCar-v0
    'MountainCar-v0': ['Push Left', 'No Push', 'Push Right'],

    # https://github.com/openai/gym/wiki/MountainCarContinuous-v0
    'MountainCarContinuous-v0': ['Push Direction'],
}

"""
Stores the labels of observation dimensions for known environments.  
"""
OBSERVATION_LABELS = {
    # gym.envs.classic_control.cartpole.CartPoleEnv
    'CartPole-v0': ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity'],

    # gym.envs.classic_control.acrobot.AcrobotEnv
    'Acrobot-v1': ['Cos(Theta 1)', 'Sin(Theta 1)', 'Cos(Theta 2)', 'Sin(Theta 2)',
                   'Theta1 Angular Velocity', 'Theta2 Angular Velocity'],

    # https://github.com/openai/gym/wiki/Pendulum-v0
    'Pendulum-v1': ['Cos(Theta)', 'Sin(Theta)', 'Angular Velocity'],

    # https://github.com/openai/gym/wiki/MountainCar-v0
    'MountainCar-v0': ['Position', 'Velocity'],

    # https://github.com/openai/gym/wiki/MountainCarContinuous-v0
    'MountainCarContinuous-v0': ['Position', 'Velocity'],
}


def is_spatial(space: gym.Space, n: int = 2) -> Tuple[bool, List[int]]:
    """
    Checks whether the given gyn space can be used to represent $N$-dimensional spatial data, with $N>1$, where the
    state or action can be represented by spatial features (layers). For example, an RGB image or the full state of a
    2D/3D game board, etc. This is true iff the space's shape has at least $N$ dimensions.
    :param gym.Space space: the space to be tested.
    :param int n: the minimum number of dimensions of the spatial features.
    :rtype: (bool, list[int])
    :return: a tuple containing a Boolean value indicating whether the space represents $N$-D spatial features, and
    the indices of the space dimensions (shape) in ascending order by dimension size.
    """
    # assume spatial if space has at least N dims
    if isinstance(space, (gym.spaces.Box, RepeatedDiscrete)) and len(space.shape) >= n:
        # assumes spatial box dims are the largest dimensions, others define multiple spatial layers...
        sorted_idxs = sorted(np.arange(len(space.shape)), key=lambda i: space.shape[i])
        return True, sorted_idxs
    return False, []  # not recognized as a spatial representation
