import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional
from gymnasium.spaces import Discrete
from ray.rllib import Policy, SampleBatch
from ixdrl import Data, MultiData
from ixdrl.data_collection.rllib.extractor import RLLibDataExtractor

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class MBMPODataExtractor(RLLibDataExtractor):
    """
    A data extractor for rllib MBMPO policies.
    """

    def __init__(self, policy: Policy, env: gym.Env, framework: str, horizon: int):
        super().__init__(policy, env, framework, horizon)

    def get_model_predictions(self, policy_info: Dict[str, np.ndarray], horizon: int = 1) -> \
            Tuple[Optional[Data], Optional[Data]]:

        observation = policy_info[SampleBatch.OBS]
        action = policy_info[SampleBatch.ACTIONS]

        # get predicted observations and rewards, predictions are deterministic, ie point predictions
        if isinstance(self.policy.action_space, Discrete):
            # one-hot-encode actions, see: ray/rllib/env/wrappers/model_vector_env.py:97
            action = action.reshape(-1)
            action = np.eye(self.policy.action_space.n)[action].astype(np.float32)

        # probe each model in the ensemble # TODO perform imagination rollout up to horizon
        td_ensemble = self.policy.dynamics_model
        orig_index = td_ensemble.sample_index
        orig_normalize = td_ensemble.normalize_data
        td_ensemble.normalize_data = False
        next_obs = []
        for i in range(td_ensemble.num_models):
            td_ensemble.sample_index = i
            next_obs.append(td_ensemble.predict_model_batches(observation, action))

        # gets actual (not predicted) rewards by probing environment
        reward = getattr(self.env, 'reward', None)
        if callable(reward):
            next_rwd = [reward(observation, action, no).reshape(-1, 1) for no in next_obs]
        else:
            next_rwd = np.empty((observation.shape[0]), 1)
            next_rwd[:] = np.nan
            next_rwd = [next_rwd for _ in next_obs]

        # set model params back
        td_ensemble.sample_index = orig_index
        td_ensemble.normalize_data = orig_normalize

        # return multiple atomic predictions
        return MultiData(next_obs), MultiData(next_rwd)
