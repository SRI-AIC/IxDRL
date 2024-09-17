import argparse
import json
import logging
import os
from functools import partial

import gymnasium as gym
import numpy as np
from gymnasium.envs import registry
from gymnasium.wrappers import TimeLimit
from ray.cloudpickle import cloudpickle
from ray.rllib.env import EnvContext
from ray.rllib.utils.from_config import from_config
from ray.tune.registry import register_env
from torch._torch_docs import merge_dicts

from ixdrl.data_collection import INTERACTION_DATA_FILE, print_stats, INTERACTION_PLOTS_FILE, \
    save_metadata
from ixdrl.data_collection.rllib import RLLibDataCollector
from ixdrl.util.cmd_line import save_args, str2bool, str2log_level
from ixdrl.util.io import create_clear_dir, save_object, load_object
from ixdrl.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Collects interaction data from policies learned using the **RLLib** toolkit.'

MBMPO_WRAPPERS_SPECS = {
    'ray.rllib.examples.env.mbmpo_env.CartPoleWrapper': 'CartPole-v0',
    'ray.rllib.examples.env.mbmpo_env.PendulumWrapper': 'Pendulum-v1',
}


def _dummy_env_creator(env_context: EnvContext,
                       env_descriptor: str) -> gym.Wrapper:
    """
    Creates a new env wrapper to store seed and correct problems with specific types of envs.
    """

    class _CustomEnv(gym.Wrapper):
        def __init__(self, _env):
            super().__init__(env)
            self.env_seed: int = None
            self._rng = None

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            if 'ALE' in self.spec.id:
                terminated |= truncated  # for ATARI envs, terminated=truncated otherwise doesn't reset
            return obs, reward, terminated, truncated, info

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            # set seed at every reset based on original seed; ensures each env has it's own seed
            if self._rng is None and seed is not None:
                self._rng = np.random.RandomState(seed)
            if self._rng is not None:
                # registers seed to be able to reproduce initial state
                self.env_seed = self._rng.randint(2 ** 31)
            self.env_seed = seed
            return super().reset(seed=self.env_seed, options=options)

    class _PyBulletEnv(_CustomEnv):
        def __init__(self, _env):
            super().__init__(_env)

        def render(self, mode='human', **kwargs):
            # returns boolean instead of empty array, see: pybullet_envs/env_bases.py:101
            r = self.env.render(mode, **kwargs)
            return self.env.isRender if mode == 'human' else r

    if 'ALE' in env_descriptor:
        # for ATARI games, there's no default limit so add it manually
        # otherwise we might get games on infinite loops
        env_context['max_num_frames_per_episode'] = 15000

    if 'Bullet' in env_descriptor:
        # PyBullet env, add wrapper
        logging.info(f'PyBullet env detected: {env_descriptor}, loading environments...')
        import pybullet_envs
        pybullet_envs.getList()
        env = gym.make(env_descriptor, **env_context)
        return _PyBulletEnv(env)
    elif '.' in env_descriptor:
        # if env directly points to a class, invoke the class
        # see: ray.rllib.agents.trainer.Trainer.setup
        logging.info(f'Class path env detected: {env_descriptor}...')
        env = from_config(env_descriptor, env_context)
    else:
        # creates an env through the gym registry
        logging.info(f'Creating registered environment: {env_descriptor}, {env_context}...')
        env = gym.make(env_descriptor, **env_context)

    if env_descriptor in MBMPO_WRAPPERS_SPECS:
        # MBMPO custom env, wrap env if time-limited
        logging.info(f'MBMPO env detected: {env_descriptor} getting spec...')
        spec = registry.spec(MBMPO_WRAPPERS_SPECS[env_descriptor])
        env.unwrapped.spec = spec
        if hasattr(spec, 'max_episode_steps'):
            # wraps the environment in a time-limit env as per the spec
            logging.info(f'Adding time limit to: {env_descriptor} ({spec.max_episode_steps} steps)...')
            env = TimeLimit(env.unwrapped, spec.max_episode_steps)

    # add normal seed storing wrapper
    return _CustomEnv(env)


def _modify_args(args: argparse.Namespace, out_dir: str):
    """
    Automatically modify some rllib evaluate script arguments to allow collecting data from the policy.
    """
    if args.render or args.video_dir is not None:
        args.video_dir = 'videos'

    args.storage_path = os.path.abspath(out_dir)
    args.out = os.path.join(out_dir, 'rllib_data.pkl')
    args.config = args.config or {}
    config = args.config
    config['train_batch_size'] = 1  # no training
    config['create_env_on_driver'] = True

    if not config.get('evaluation_num_workers'):
        config['evaluation_num_workers'] = config.get('num_workers', 0)
    if not config.get('evaluation_duration'):
        config['evaluation_duration'] = 1

    config['evaluation_interval'] = 1
    config['rollout_fragment_length'] = 'auto'
    config['render_env'] = args.render

    config['explore'] = False  # no explore
    config['simple_optimizer'] = True  # avoid multi-GPU
    if args.run == 'MBMPO':
        config['dynamics_model'] = dict(train_epochs=0, batch_size=1)  # no training..

    # load config, based on ray.rllib.evaluate.run, to get correct framework and env info
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.abspath(os.path.join(config_dir, 'params.pkl'))
    if not os.path.exists(config_path):
        config_path = os.path.abspath(os.path.join(config_dir, '..', 'params.pkl'))
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            load_config = cloudpickle.load(f)
            args.config = merge_dicts(load_config, config)
            config = args.config
        logging.info(f'Found config file at: {config_path}, setting:'
                     f'\n\tframework: "{config.get("framework", "tf")}",'
                     f'\n\tenvironment: "{config["env"]}"')
        args.env = config['env']
        args.framework = config['framework']
        if 'env_config' in config:
            args.config['env_config'] = config['env_config']
            if args.render:
                args.config['env_config']['render_mode'] = 'rgb_array'

        if 'sgd_minibatch_size' in config:
            args.config['sgd_minibatch_size'] = 1  # no training
    else:
        logging.info(f'Config file does not exist: {config_path}')

    # add wrapper
    register_env(args.env, partial(_dummy_env_creator, env_descriptor=args.env))


def create_parser(parser_creator=None):
    # copied RLLib args from previous version that used argparse
    # see: https://github.com/ray-project/ray/blob/releases/2.1.0/rllib/evaluate.py#L48

    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__description__,
    )

    parser.add_argument(
        "checkpoint",
        type=str,
        nargs="?",
        help="(Optional) checkpoint from which to roll out. "
             "If none given, will use an initial (untrained) Trainer.",
    )

    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
             "of a built-on algorithm (e.g. RLlib's `DQN` or `PPO`), or a "
             "user-defined trainable function or class registered in the "
             "tune registry.",
    )
    required_named.add_argument(
        "--env",
        type=str,
        help="The environment specifier to use. This could be an openAI gym "
             "specifier (e.g. `CartPole-v0`) or a full class-path (e.g. "
             "`ray.rllib.examples.env.simple_corridor.SimpleCorridor`).",
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Run ray in local mode for easier debugging.",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment while evaluating."
    )
    # Deprecated: Use --render, instead.
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Deprecated! Rendering is off by default now. "
             "Use `--render` to enable.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Specifies the directory into which videos of all episode "
             "rollouts will be stored.",
    )
    parser.add_argument(
        "--steps",
        default=10000,
        help="Number of timesteps to roll out. Rollout will also stop if "
             "`--episodes` limit is reached first. A value of 0 means no "
             "limitation on the number of timesteps run.",
    )
    parser.add_argument(
        "--episodes",
        default=0,
        help="Number of complete episodes to roll out. Rollout will also stop "
             "if `--steps` (timesteps) limit is reached first. A value of 0 means "
             "no limitation on the number of episodes run.",
    )
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
             "Gets merged with loaded configuration from checkpoint file and "
             "`evaluation_config` settings therein.",
    )
    parser.add_argument(
        "--save-info",
        default=False,
        action="store_true",
        help="Save the info field generated by the step() method, "
             "as well as the action, observations, rewards and done fields.",
    )
    parser.add_argument(
        "--use-shelve",
        default=False,
        action="store_true",
        help="Save rollouts into a python shelf file (will save each episode "
             "as it is generated). An output filename must be set using --out.",
    )
    parser.add_argument(
        "--track-progress",
        default=False,
        action="store_true",
        help="Write progress to a temporary file (updated "
             "after each episode). An output filename must be set using --out; "
             "the progress file will live in the same folder.",
    )
    return parser


def main():
    parser = create_parser()

    # add custom args and parse args
    parser.add_argument('--output', '-o', type=str, default=None, required=True,
                        help='The path to the directory in which to save the collected interaction data.')
    parser.add_argument('--stats-only', '-so', type=str2bool,
                        help='Whether to use previously-collected data and print stats only.'
                             'If `True` but the data file cannot be found, then will still collect the data.')
    parser.add_argument('--img-format', '-if', type=str, default='pdf', help='The format of image files.')
    parser.add_argument('--labels-file', '-lf', type=str, default=None,
                        help='The path to a json file containing specifications for the action and observation '
                             'labels of custom environments.')
    parser.add_argument('--fps', type=int, default=30,
                        help='The frames per second rate used to save the episode videos.')
    parser.add_argument('--clear', '-c', type=str2bool, help='Clear output directories before generating results.')
    parser.add_argument('--verbosity', '-v', type=str2log_level, default=0, help='Verbosity level.')
    args = parser.parse_args()

    # checks output dir and log file
    out_dir = args.output
    create_clear_dir(out_dir, args.clear and not args.stats_only)
    log_file = os.path.abspath(os.path.join(out_dir, 'data_collection.log'))
    change_log_handler(log_file, append=os.path.isfile(log_file), level=args.verbosity)
    save_args(args, os.path.join(out_dir, 'args.json'))

    # loads or generates data
    data_file = os.path.join(out_dir, INTERACTION_DATA_FILE)
    if args.stats_only and os.path.isfile(data_file):
        # try to load existing data file
        logging.info(f'Loading interaction data from {data_file}...')
        data = load_object(data_file)
        data = {r_id: rollout for i, (r_id, rollout) in enumerate(data.items())}
        logging.info(f'Loaded {len(data)} rollouts')
    else:
        # modify arguments to allow for data collection
        _modify_args(args, out_dir)

        # create and run RLLib data collector
        collector = RLLibDataCollector(args, out_dir)
        data = collector.collect_data(int(args.episodes))
        logging.info(f'Finished collecting interaction data from {len(data)} rollouts')

        # save consolidated results
        logging.info(f'Saving consolidated results to {data_file}...')
        save_object(data, data_file, compress_gzip=True)
        collector.remove_temp_data()  # delete temp dir, not needed

    # saves metadata
    file_path = os.path.join(out_dir, 'metadata.json')
    save_metadata(data, file_path)
    logging.info(f'Saved rollouts metadata to {file_path}')

    # print stats
    stats_dir = os.path.join(out_dir, 'stats')
    create_clear_dir(stats_dir, args.clear)
    logging.info(f'Saving interaction data statistics to {stats_dir}...')
    figures = print_stats(data, stats_dir, args.img_format)
    figs_file = os.path.join(out_dir, INTERACTION_PLOTS_FILE)
    save_object(figures, figs_file, compress_gzip=True)
    logging.info(f'Saved all plotly figures to "{figs_file}"')

    logging.info('Done!')


if __name__ == "__main__":
    main()
