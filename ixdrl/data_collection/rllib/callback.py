import json
import logging
import numpy as np
import os
import shutil
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from ray.rllib import RolloutWorker, BaseEnv, SampleBatch, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from typing import Optional, List, Dict, Union

from ixdrl import InteractionData, Rollout
from ixdrl.data_collection.rllib.ddpg import DDPGDataExtractor
from ixdrl.data_collection.rllib.dqn import DQNDataExtractor
from ixdrl.data_collection.rllib.extractor import RLLibDataExtractor
from ixdrl.data_collection.rllib.mbmpo import MBMPODataExtractor
from ixdrl.data_collection.rllib.sac import SACDataExtractor
from ixdrl.util.gym import get_action_labels, get_observation_labels, ACTION_LABELS, OBSERVATION_LABELS
from ixdrl.util.io import save_object, get_file_changed_extension

__author__ = 'Pedro Sequeira, Sam Showalter'
__email__ = 'pedro.sequeira@sri.com'

DISCOUNT_PARAM = 'gamma'


class DataCollectionCallback(DefaultCallbacks):
    """
    An rllib callback used to collect interaction data during rollouts.
    """

    def __init__(self,
                 output_dir: str,
                 video_dir: str,
                 model: str,
                 framework: str,
                 env_id: str,
                 labels_file: str,
                 record_video: bool,
                 fps: int):
        """
        Creates a new data collection callback.
        :param str output_dir: the path to the directory in which to save (temporary) results.
        :param str video_dir: the path to the directory in which to save videos.
        :param str model: the RL agent's model/algorithm type, i.e., DQN, A2C, etc.
        :param str framework: the DL framework, either tf, tf2 or torch.
        :param str env_id: the id of the environment for which data is going to be collected.
        :param str labels_file: the path to a json file containing specifications for the action and observation.
        :param bool record_video: whether to record videos of each episode.
        :param int fps: the frames per second rate of recorded episode videos.
        labels of custom environments.
        """
        super().__init__()
        self.output_dir = output_dir
        self.model = model
        self.framework = framework
        self.env_id = env_id
        self.video_dir = video_dir
        self.record_video = record_video
        self.fps = fps

        # loads labels file, if any
        if labels_file is not None and os.path.isfile(labels_file):
            with open(labels_file, 'r') as fp:
                labels = json.load(fp)
            logging.info(f'Loaded {len(labels["actions"])} extra action and observation spaces labels '
                         f'from: {labels_file}')
            ACTION_LABELS.update(labels['actions'])
            OBSERVATION_LABELS.update(labels['observations'])

        # episode data
        self.episode_num = -1
        self.video_recorders: Dict[int, VideoRecorder] = {}  # to store the video recorders
        self.extractors: Dict[int, RLLibDataExtractor] = {}  # to store the data extractors
        self.datapoints: Dict[int, List[InteractionData]] = {}  # to store the datapoint info over time
        self.obs: Dict[int, Optional[np.ndarray]] = {}
        self.rwd: Dict[int, float] = {}

    def on_episode_created(self,
                           *,
                           worker: "RolloutWorker",
                           base_env: BaseEnv,
                           policies: Dict[PolicyID, Policy],
                           env_index: int,
                           episode: Union[Episode, EpisodeV2],
                           **kwargs) -> None:

        # set up video recorder (use temp dir since recorders are created for "dummy" episodes by RLLib)
        if self.record_video:
            episode_id = episode.episode_id
            env = base_env.get_sub_environments()[env_index]
            env.metadata['render_fps'] = self.fps
            recorder = VideoRecorder(env=env,
                                     path=os.path.join(self.output_dir, f'{episode_id}.mp4'),
                                     metadata={'episode_id': episode_id, 'fps': self.fps})
            recorder.capture_frame()
            self.video_recorders[episode_id] = recorder

    def on_episode_start(self,
                         *,
                         worker: RolloutWorker,
                         base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: EpisodeV2,
                         env_index: Optional[int] = None,
                         **kwargs) -> None:

        # reset data, increment counters
        self.episode_num += 1
        logging.info(f'[Ep. {episode.episode_id}]: (#{self.episode_num}) started collecting data...')

        # create policy extractor for this step # TODO get pred horizon from config?
        policy = worker.policy_map[episode.policy_for()]
        env = base_env.get_sub_environments()[env_index]
        self.extractors[episode.episode_id] = self._get_data_extractor(policy, env)

        self.datapoints[episode.episode_id] = []
        self.obs[episode.episode_id] = None  # we need to get first obs from samplers
        self.rwd[episode.episode_id] = 0  # rwd is associated with (prev) obs

        # initialize a dummy extractor such that extra data can be fetched from policy if needed
        policy = worker.policy_map[episode.policy_for()]
        self._get_data_extractor(policy, base_env.get_sub_environments()[env_index])

    def on_episode_step(self,
                        *,
                        worker: RolloutWorker,
                        base_env: BaseEnv,
                        policies: Optional[Dict[PolicyID, Policy]] = None,
                        episode: EpisodeV2,
                        env_index: Optional[int] = None,
                        **kwargs) -> None:

        episode_id = episode.episode_id
        buffers = episode._agent_collectors[_DUMMY_AGENT_ID].buffers
        policy = worker.policy_map[episode.policy_for()]
        if self.obs[episode_id] is None:
            self.obs[episode_id] = buffers[SampleBatch.OBS][0][0]  # gets first observation
        self.video_recorders[episode_id].capture_frame()

        last_info = {key: val[0][-1] for key, val in buffers.items()}  # get last collected policy information

        self.datapoints[episode_id].append(
            self.extractors[episode_id].get_interaction_datapoint(
                self.obs[episode_id][np.newaxis, ...],
                np.array([[last_info[SampleBatch.ACTIONS]]]),
                np.array([[self.rwd[episode_id]]]),
                policy.get_initial_state(),
                np.array([last_info[SampleBatch.T]])
            )
        )

        # TODO just for testing batch mode, test with current and next obs
        idps = self.extractors[episode_id].get_interaction_datapoint(
            np.array([self.obs[episode_id], last_info[SampleBatch.OBS]]),
            None,
            np.full((2, 1), self.rwd[episode_id]),
            policy.get_initial_state(),
            np.full(2, last_info[SampleBatch.T]))

        # store new data
        self.obs[episode_id] = np.asarray(last_info[SampleBatch.OBS])
        self.rwd[episode_id] = last_info[SampleBatch.REWARDS]

    def on_episode_end(self,
                       *,
                       worker: RolloutWorker,
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: EpisodeV2,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:

        # save rollout data to pickle file for later retrieval
        config = worker.policy_map[episode.policy_for()].config
        gamma = config[DISCOUNT_PARAM] if DISCOUNT_PARAM in config else None
        env = base_env.get_sub_environments()[env_index]
        action_labels = get_action_labels(env)
        obs_dim_labels = get_observation_labels(env)
        episode_id = episode.episode_id

        # close video recorder
        video_recorder = self.video_recorders[episode_id]
        del video_recorder.recorded_frames[-1]  # remove last frame since observations are from previous step
        video_recorder.metadata['steps'] = len(self.datapoints[episode_id])
        video_recorder.metadata['frames'] = len(video_recorder.recorded_frames)
        video_recorder.close()

        # move files to video dir
        video_path = os.path.join(self.video_dir, os.path.basename(video_recorder.path))
        shutil.move(video_recorder.path, video_path)
        shutil.move(get_file_changed_extension(video_recorder.path, ext='meta.json'),
                    get_file_changed_extension(video_path, ext='meta.json'))

        rollout = Rollout(rollout_id=str(episode_id),
                          data=self.datapoints[episode_id],
                          observation_space=env.observation_space,
                          action_space=env.action_space,
                          rwd_range=env.reward_range,
                          discount=gamma,
                          action_labels=action_labels,
                          observation_labels=obs_dim_labels,
                          video_file=video_path,
                          env_id=self.env_id,
                          seed=env.env_seed)

        if hasattr(env, 'lives'):
            rollout.data.user_data['lives'] = env.lives + 1  # for ATARI envs with lives, set it here
        data_file = os.path.join(self.output_dir, f'{episode_id}.pkl.gz')
        save_object(rollout, data_file, compress_gzip=True)
        logging.info(f'[Ep. {episode_id}]: collected interaction data for {len(rollout.data.timesteps)} steps.')
        logging.info(f'[Ep. {episode_id}]: saved data to {data_file}.')

        # remove data
        del self.video_recorders[episode_id]
        del self.extractors[episode_id]
        del self.datapoints[episode_id]
        del self.obs[episode_id]
        del self.rwd[episode_id]

    def _get_data_extractor(self, policy, env):
        # chooses extractor according to model
        extractor = RLLibDataExtractor
        if self.model in {'DQN'}:
            extractor = DQNDataExtractor
        elif self.model == 'SAC':
            extractor = SACDataExtractor
        elif self.model in {'DDPG', 'APEX_DDPG', 'TD3'}:
            extractor = DDPGDataExtractor
        elif self.model == 'MBMPO':
            extractor = MBMPODataExtractor

        return extractor(policy, env, self.framework, horizon=1)
