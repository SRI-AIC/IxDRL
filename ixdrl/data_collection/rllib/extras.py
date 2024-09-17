from gymnasium.envs.registration import EnvSpec
from pybullet_envs_gymnasium.gym_locomotion_envs import HalfCheetahBulletEnv

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class HalfCheetahEnv(HalfCheetahBulletEnv):
    def __init__(self, env_config=None):
        super().__init__(render_mode=env_config['render_mode'] if env_config and 'render_mode' in env_config else None)
        max_steps = env_config['max_episode_steps'] if env_config and 'max_episode_steps' in env_config else 1000
        self.spec = EnvSpec(id='HalfCheetahEnv', max_episode_steps=max_steps)
        self._max_episode_steps = max_steps
        self.steps = 0

    def step(self, a):
        self.steps += 1
        return super().step(a)
