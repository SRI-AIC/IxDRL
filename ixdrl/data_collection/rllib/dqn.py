import gymnasium as gym
import numpy as np
from typing import Dict, Optional
from scipy.special import softmax
from ray.rllib import Policy, SampleBatch
from ray.rllib.utils import try_import_tf
from ixdrl.types import Data, Distribution, CategoricalDistribution, AtomicData
from ixdrl.data_collection.rllib.extractor import RLLibDataExtractor

__author__ = 'Pedro Sequeira, Sam Showalter'
__email__ = 'pedro.sequeira@sri.com'

Q_VALUES = 'q_values'
Q_VALUES_DIST = 'q_values_dist'

tf1, *_ = try_import_tf()


class DQNDataExtractor(RLLibDataExtractor):
    """
    A data extractor for rllib DQN policies.
    """

    def __init__(self, policy: Policy, env: gym.Env, framework: str, horizon: int):
        super().__init__(policy, env, framework, horizon)

        # add extra information to be fetched if Distributional DQN
        if (self.policy.config['num_atoms'] > 1 and self.framework == 'tf'
                and Q_VALUES_DIST not in self.policy._extra_action_fetches):
            self.policy._extra_action_fetches[Q_VALUES_DIST] = 'default_policy_wk1/model_1/lambda_1/Softmax:0'

        self.action_dist = None
        self.action_ph = None
        self.input_ph = None

    def get_value(self, policy_info: Dict[str, np.ndarray]) -> Optional[Data]:
        # use max Q since value_function doesn't seem to return "the" value estimate
        q_values = self.get_action_values(policy_info)
        if q_values is None:
            return None
        if isinstance(q_values, AtomicData):
            # if atomic data then get Q max
            return np.max(q_values, axis=1, keepdims=True)
        if isinstance(q_values, CategoricalDistribution):
            # if categorical (distDQN), get value distribution for action with max mean value
            values = np.sum(q_values.probs * q_values.support, axis=-1)  # shape: (batch, n_actions)
            max_action = np.argmax(values, axis=-1)  # shape: (batch,)
            batch_idx = np.arange(values.shape[0])
            return CategoricalDistribution(
                q_values.probs[batch_idx, max_action], q_values.support[batch_idx, max_action])
        raise ValueError(f'Cannot get value from Q-values of unsupported type: {q_values}')

    def get_action_values(self, policy_info: Dict[str, np.ndarray]) -> Optional[Data]:
        # standard DQN, just return the Q-values stored in policy info, shape: (batch, action_shape)
        q_values = policy_info[Q_VALUES] if Q_VALUES in policy_info else None

        config = self.policy.config
        if config['num_atoms'] > 1:
            # distributional DQN, add the discrete distribution over Q-values instead of sum over them
            # see ray.rllib.agents.dqn.distributional_q_tf_model.DistributionalQTFModel.get_q_value_distributions
            # and ray.rllib.agents.dqn.dqn_torch_model.DQNTorchModel.get_q_value_distributions

            if Q_VALUES_DIST in policy_info:
                # tf1: distribution available in dict
                dist = policy_info[Q_VALUES_DIST]

                # create support vector, see: ray/rllib/agents/dqn/distributional_q_tf_model.py:118
                z = np.arange(config['num_atoms'])
                z = config['v_min'] + z * (config['v_max'] - config['v_min']) / (config['num_atoms'] - 1)

            else:
                # otherwise we need to get the information (tf2, tfe, torch)
                # get last policy model output (assumes last call was performed by get_interaction_datapoint)
                model_out = self.policy.model._last_output
                action_scores, z, support_logits_per_action, logits, dist = \
                    self.policy.model.get_q_value_distributions(model_out)

            dist = self._to_numpy(dist)  # the probabilities for each atom
            support = np.empty_like(dist)  # the support / set of atoms (same for all actions)
            support[..., :] = self._to_numpy(z)
            return CategoricalDistribution(dist, support)

        return q_values

    def get_action_dist(self, policy_info: Dict[str, np.ndarray]) -> Optional[Distribution]:
        # return softmax over Q-values
        return CategoricalDistribution(softmax(policy_info[Q_VALUES], axis=1)) if Q_VALUES in policy_info else None

    def get_action_prob(self, policy_info: Dict[str, np.ndarray]) -> Optional[AtomicData]:
        if SampleBatch.ACTION_DIST_INPUTS not in policy_info:
            # last resort, just send what's in the policy info
            return policy_info[SampleBatch.ACTION_PROB] if SampleBatch.ACTION_PROB in policy_info else None

        # otherwise, probe action distribution for prob of actual action(s)
        action = policy_info[SampleBatch.ACTIONS][:, 0]
        if self.framework == 'torch':
            import torch
            action = torch.from_numpy(action)

        action_dist_inputs = policy_info[SampleBatch.ACTION_DIST_INPUTS]
        if self.framework == 'tf':
            if self.action_dist is None:
                # create action log prob operation
                self.input_ph = tf1.placeholder(tf1.float32, shape=(None, action_dist_inputs.shape[1]))
                self.action_ph = tf1.placeholder(tf1.int32, shape=(None,))
                self.action_dist = self.policy.dist_class(self.input_ph, self.policy.model).logp(self.action_ph)

            # get operation result from current dist input and selected action
            with tf1.Session(graph=self.action_dist.graph) as sess:
                log_p = sess.run(
                    self.action_dist, feed_dict={self.input_ph: action_dist_inputs, self.action_ph: action})
        else:
            # otherwise, sample action log prob directly
            action_dist = self.policy.dist_class(action_dist_inputs, self.policy.model)
            log_p = action_dist.logp(action)

        prob = np.clip(np.exp(self._to_numpy(log_p)), 0, 1)
        if len(prob.shape) < 2:
            prob = prob.reshape(-1, 1)
        return prob
