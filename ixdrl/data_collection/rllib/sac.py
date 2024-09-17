import gymnasium as gym
import numpy as np
from typing import Dict, Optional
from ray.rllib import Policy, SampleBatch
from ixdrl import Data
from ixdrl.data_collection.rllib.extractor import RLLibDataExtractor

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class SACDataExtractor(RLLibDataExtractor):
    """
    A data extractor for rllib SAC policies.
    """

    def __init__(self, policy: Policy, env: gym.Env, framework: str, horizon: int):
        super().__init__(policy, env, framework, horizon)

    def get_value(self, policy_info: Dict[str, np.ndarray]) -> Optional[Data]:

        if isinstance(self.policy.action_space, gym.spaces.Discrete):
            # discrete action space, get max of Q-values
            q_values = self.get_action_values(policy_info)
            return np.max(q_values, axis=1, keepdims=True)

        # for SAC use model's Q-value estimator for given observation(s) and action(s), based on
        # ray.rllib.agents.sac.sac_torch_policy.actor_critic_loss and
        # ray.rllib.agents.sac.sac_tf_policy.sac_actor_critic_loss
        act = policy_info[SampleBatch.ACTIONS]
        if self.framework == 'tf':
            with self.policy.model.graph.as_default():
                # see ray.rllib.agents.sac.sac_tf_model.SACTFModel.get_q_values
                model_out, _ = self.policy.model({SampleBatch.OBS: policy_info[SampleBatch.OBS]})
                val = self.policy.model.get_q_values(model_out, act)
        else:
            model_out, _ = self.policy.model({SampleBatch.OBS: policy_info[SampleBatch.OBS]})
            if self.framework == 'torch':
                import torch
                act = torch.from_numpy(act)
                model_out = torch.from_numpy(model_out)
            val = self.policy.model.get_q_values(model_out, act)

        return self._to_numpy(val)

    def get_action_values(self, policy_info: Dict[str, np.ndarray]) -> Optional[Data]:
        if not isinstance(self.policy.action_space, gym.spaces.Discrete):
            # no action values if not discrete action space
            return None

        # for SAC use model's Q-value estimator for given observation(s), based on
        # ray.rllib.agents.sac.sac_torch_policy.actor_critic_loss and
        # ray.rllib.agents.sac.sac_tf_policy.sac_actor_critic_loss
        if self.framework == 'tf':
            with self.policy.model.graph.as_default():
                # see ray.rllib.agents.sac.sac_tf_model.SACTFModel.get_q_values
                model_out, _ = self.policy.model({SampleBatch.OBS: policy_info[SampleBatch.OBS]})
                vals = self.policy.model.get_q_values(model_out)
        else:
            model_out, _ = self.policy.model({SampleBatch.OBS: policy_info[SampleBatch.OBS]})
            if self.framework == 'torch':
                import torch
                model_out = torch.from_numpy(model_out)
            vals = self.policy.model.get_q_values(model_out)

        return self._to_numpy(vals)
