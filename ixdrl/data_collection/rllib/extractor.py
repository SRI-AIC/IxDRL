import gymnasium as gym
import numbers
import numpy as np
from gymnasium import spaces as spaces
from ray.rllib import Policy, SampleBatch
from ray.rllib.utils import try_import_tf
from scipy.special import softmax
from typing import Dict, List, Tuple, Optional

from ixdrl import InteractionData, Data, Distribution, AtomicData, CategoricalDistribution, \
    NormalDistribution

tf1, *_ = try_import_tf()

__author__ = 'Pedro Sequeira, Sam Showalter'
__email__ = 'pedro.sequeira@sri.com'


class RLLibDataExtractor(object):
    """
    Represents extractors of interaction datapoints for rllib agents.
    This class contains the data extractors for generic rllib agents, while sub-classes implement extractors for
    specific algorithms.
    """

    def __init__(self, policy: Policy, env: gym.Env, framework: str, horizon: int):
        """
        Creates a new rllib interaction data extractor.
        :param Policy policy: the agent's policy model from which to extract the interaction data.
        :param gym.Env env: the environment in which the agent is acting.
        :param str framework: the DL framework, either tf, tf2 or torch.
        :param int horizon: the horizon at which the next observation and reward predictions are to be made
        (only for model-based RL algorithms).
        """
        self.policy = policy
        self.env = env
        self.framework = framework
        self.horizon = horizon

    def get_interaction_datapoint(self,
                                  obs: np.ndarray,
                                  action: Optional[np.ndarray],
                                  reward: np.ndarray,
                                  prev_state_out: List[np.ndarray],
                                  timestep: np.ndarray) -> InteractionData:
        """
        Gets an interaction datapoint from data by probing the agent's policy on the given observations.
        This can be used to recover the policy information produced for some observation during an episode, or to probe
        an agent on arbitrary observations.
        :param np.ndarray obs: the observations for which we want the interaction data, an array of shape
        (batch_size, *observation_space.shape).
        :param np.ndarray action: the actions that the agent executed given the observations, to be stored in the
        datapoint. If `None`, the stored actions will be sampled from the policy. An array of shape (batch_size, *action_space.shape).
        :param np.ndarray reward: the rewards received by the agent from the environment, an array of shape
        (batch_size, 1).
        :param list[np.ndarray] prev_state_out: the previous RNN state(s) outputs, where each item is an array of shape
        (batch_size, state_size).
        :param np.ndarray timestep: the timesteps in which the observations were taken, an array of shape (batch_size, ).
        :rtype: InteractionData
        :return: the interaction datapoint extracted by this extractor for the given observation.
        """
        # get info by probing policy, a tuple containing:
        # 0: the action(s) taken by the agent
        # 1: the RNN state(s) outputs, if any, with shape (batch_size, state_size)
        # 2: dictionary of policy info batches, which is used to extract the necessary interaction data
        input_dict = dict(obs=obs, _is_training=False)
        for i, state_in in enumerate(prev_state_out):
            input_dict[f'state_in_{i}'] = state_in  # add rnn states in the correct format # TODO check this
        _action, state_out, policy_info = self.policy.compute_actions_from_input_dict(
            SampleBatch(input_dict), explore=False)

        # store data in policy info object
        obs = self._to_numpy(obs)
        policy_info[SampleBatch.OBS] = obs
        policy_info[SampleBatch.ACTIONS] = action or _action  # if not provided, use the computed actions from policy
        policy_info[SampleBatch.REWARDS] = reward

        # get info in batch format
        def _to_batch(data):
            return data[..., np.newaxis] if len(data.shape) == 1 else data  # assume 1-dim data

        for k, v in policy_info.items():
            if v is None:
                continue
            if isinstance(v, list):
                for i in range(len(v)):
                    policy_info[k][i] = _to_batch(v[i])  # process each item in the list
            policy_info[k] = _to_batch(v)

        # get the necessary information from the policy in batch mode
        value = self.get_value(policy_info)
        action_dist = self.get_action_dist(policy_info)
        action_values = self.get_action_values(policy_info)
        action_prob = self.get_action_prob(policy_info)
        pred_obs, pred_rwd = self.get_model_predictions(policy_info, self.horizon)

        # create interaction datapoint
        return InteractionData(
            observation=policy_info[SampleBatch.OBS],
            action=policy_info[SampleBatch.ACTIONS],
            reward=policy_info[SampleBatch.REWARDS],
            value=value,
            action_dist=action_dist,
            action_values=action_values,
            action_prob=action_prob,
            pred_obs=pred_obs,
            pred_rwd=pred_rwd,
            timesteps=timestep.tolist(),
            user_data=dict(state_out=state_out)
        )

    def get_value(self, policy_info: Dict[str, np.ndarray]) -> Optional[Data]:
        """
        Get the value associated with the observation for which the action was taken.
        :param dict[str, np.ndarray] policy_info: the dictionary containing the results from the policy's probing.
        :rtype: np.ndarray or None
        :returns: the state value, an array of shape (batch_size, 1).
        """
        if SampleBatch.VF_PREDS in policy_info:
            # return policy's value prediction if available
            return self._to_numpy(policy_info[SampleBatch.VF_PREDS])

        # otherwise fetch via value_function which provides value estimate from last forward pass, i.e.,
        # for probing the value for certain observation(s), it assumes the forward pass to the model was done, e.g.,
        # via compute_actions_from_input_dict
        try:
            val = self.policy.model.value_function()
            return self._to_numpy(val)
        except NotImplementedError:
            return None

    def get_action_dist(self, policy_info: Dict[str, np.ndarray]) -> Optional[Distribution]:
        """
        Gets the action distribution for the current state.
        :param dict[str, np.ndarray] policy_info: the dictionary containing the results from the policy's probing.
        :rtype: np.ndarray or None
        :returns: the action distribution, an array of shape (batch_size, 2) for continuous actions, parameterizing a
        Gaussian, or (batch_size, action_space.n) for discrete action spaces.
        """
        if SampleBatch.ACTION_DIST_INPUTS not in policy_info:
            return None
        dist_inputs = self._to_numpy(policy_info[SampleBatch.ACTION_DIST_INPUTS])

        # check action space and return distribution accordingly
        # general choice: ray.rllib.models.catalog.ModelCatalog.get_action_dist
        action_space = self.policy.action_space
        if isinstance(action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            # gets softmax from action logits
            return CategoricalDistribution(softmax(dist_inputs, axis=-1))

        if isinstance(action_space, spaces.Box):
            # assume Gaussian, divide dists (first half the means, second half the log standard deviations)
            # expected input shape: (batch_size, prod(action_shape)*2)
            if dist_inputs.shape[1] != np.prod(action_space.shape) * 2:
                return None  # mal-formed array?
            mean, log_std = np.split(dist_inputs, 2, axis=-1)
            return NormalDistribution(mean, np.exp(log_std))

        return None

    def get_action_prob(self, policy_info: Dict[str, np.ndarray]) -> Optional[AtomicData]:
        """
        Gets the probability associated with the executed action.
        :param dict[str, np.ndarray] policy_info: the dictionary containing the results from the policy's probing.
        :rtype: np.ndarray or None
        :returns: the action probability, an array of shape (batch_size, 1).
        """

        if SampleBatch.ACTION_DIST_INPUTS not in policy_info:
            # last resort, just send what's in the policy info
            return policy_info[SampleBatch.ACTION_PROB] if SampleBatch.ACTION_PROB in policy_info else None

        # otherwise probe action distribution for prob of actual action(s)
        action_dist = self.policy.dist_class(policy_info[SampleBatch.ACTION_DIST_INPUTS], self.policy.model)
        action = policy_info[SampleBatch.ACTIONS]

        if self.framework == 'torch':
            import torch
            action = torch.from_numpy(action)
        if hasattr(action_dist, '_squash'):
            action = action_dist._squash(action)

        prob = np.clip(np.exp(self._to_numpy(action_dist.logp(action))), 0, 1)
        if len(prob.shape) < 2:
            prob = prob.reshape(-1, 1)
        return prob

    def get_action_values(self, policy_info: Dict[str, np.ndarray]) -> Optional[Data]:
        """
        Gets the value associated with each action given the observation.
        :param dict[str, np.ndarray] policy_info: the dictionary containing the results from the policy's probing.
        :rtype: np.ndarray or None
        :returns: the action values, an array of shape (batch_size, action_space.n).
        """
        return None

    def get_model_predictions(self, policy_info: Dict[str, np.ndarray], horizon: int = 1) -> \
            Tuple[Optional[Data], Optional[Data]]:
        """
        Gets the observation(s) and reward(s) predicted by the agent's dynamics model given the previous observation
        and executed action.
        :param dict[str, np.ndarray] policy_info: the dictionary containing the results from the policy's probing.
        :param int horizon: the horizon at which the prediction is to be made.
        :rtype: (np.ndarray, np.ndarray, bool)
        :returns: a tuple with the the predicted observation(s), an array of shape (batch_size, num_predictions,
        state_space.shape, 2/observation_space), the reward(s), an array of shape (batch_size, num_predictions, 1), and
        a boolean value indicating whether the predictions are point predictions or distribution params.
        Might correspond to multiple multivariate Gaussian distributions over observations, or multiple predicted
        observations.
        """
        return None, None

    def _to_numpy(self, tensor) -> np.ndarray:
        """
        Utility method to convert tensors to numpy arrays.
        :param tensor: the tensor to be converted.
        :rtype: np.ndarray
        :return: the converted numpy array
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        if isinstance(tensor, numbers.Number):
            return np.array([tensor])  # assume batch mode
        if self.framework == 'tf':
            return tensor.eval(session=tf1.Session(graph=tensor.graph))
        if self.framework == 'tf2':
            return tensor.numpy()
        if self.framework == 'torch':
            return tensor.detach().cpu().numpy()
        raise NotImplementedError(f'Cannot extract data from policy using framework "{self.framework}"')
