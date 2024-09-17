import gymnasium as gym
import numpy as np
from typing import Dict, Optional
from ray.rllib import Policy, SampleBatch
from ixdrl import Data, Distribution
from ixdrl.data_collection.rllib.extractor import RLLibDataExtractor

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class DDPGDataExtractor(RLLibDataExtractor):
    """
    A data extractor for rllib DDPG policies.
    """

    def __init__(self, policy: Policy, env: gym.Env, framework: str, horizon: int):
        super().__init__(policy, env, framework, horizon)

    def get_value(self, policy_info: Dict[str, np.ndarray]) -> Optional[Data]:

        # for DDPG use model's Q-value estimator for given observation(s) and action(s)
        # based on ray.rllib.agents.ddpg.ddpg_torch_policy.ddpg_actor_critic_loss, and
        # ray.rllib.agents.ddpg.ddpg_tf_policy.ddpg_actor_critic_loss
        act = policy_info[SampleBatch.ACTIONS]
        if self.framework == 'tf':
            with self.policy.model.graph.as_default():
                # see ray.rllib.agents.ddpg.ddpg_tf_model.DDPGTFModel.get_q_values
                model_out, _ = self.policy.model({SampleBatch.OBS: policy_info[SampleBatch.OBS]})
                val = self.policy.model.get_q_values(model_out, act)
        else:
            if self.framework == 'torch':
                import torch
                act = torch.from_numpy(act)
            model_out = self.policy.model._last_output
            val = self.policy.model.get_q_values(model_out, act)

        return self._to_numpy(val)

    def get_action_dist(self, policy_info: Dict[str, np.ndarray]) -> Optional[Distribution]:
        # there is no action distribution since the policy is deterministic, i.e., the output of the distribution
        # object *is* the action, so return None to avoid misinterpretations downstream
        return None
