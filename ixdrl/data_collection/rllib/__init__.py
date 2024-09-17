from collections import OrderedDict

import argparse
import copy
import logging
import os
import ray
import shutil
import tqdm
from pathlib import Path
from ray.rllib.evaluate import RolloutSaver, rollout
from ray.train import Checkpoint
from ray.train._internal.session import _TrainingResult
from ray.tune.registry import get_trainable_cls

from ixdrl import Rollout, Rollouts
from ixdrl.data_collection import InteractionDataCollector
from ixdrl.data_collection.rllib.callback import DataCollectionCallback
from ixdrl.util.io import get_files_with_extension, load_object, create_clear_dir

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

ALGORITHM = 'DUMMY_ALGORITHM'


class RLLibDataCollector(InteractionDataCollector):
    """
    An interaction data collector for the RLlib framework https://docs.ray.io/en/latest/rllib.html.
    """

    def __init__(self, args: argparse.Namespace, output_dir: str):
        """
        Creates a new interaction data collector.
        :param argparse.Namespace args: the necessary arguments to spawn rllib evaluator.
        :param str output_dir: the path to the output directory in which to save results.
        """
        assert args.run is not None, '"run" argument has to be provided in order to create proper extractor callback!'
        assert args.framework is not None, \
            '"framework" has to be defined in the config in order to create proper extractor!'

        # copied from ray/rllib/evaluate.py:177
        # --use_shelve w/o --out option.
        if args.use_shelve and not args.out:
            raise ValueError('If you set --use-shelve, you must provide an output file via --out as well!')
        # --track-progress w/o --out option.
        if args.track_progress and not args.out:
            raise ValueError('If you set --track-progress, you must provide an output file via --out as well!')

        self.args = copy.deepcopy(args)
        self.args.config = args.config or {}
        self.temp_dir = os.path.join(output_dir, 'ep_data')  # temporary directory
        create_clear_dir(self.temp_dir, clear=False)
        video_dir = os.path.join(output_dir, 'videos')
        create_clear_dir(video_dir, clear=False)

        # register extractor callback
        self.callback = lambda: DataCollectionCallback(
            output_dir=self.temp_dir,
            video_dir=video_dir,
            model=args.run,
            framework=args.framework,
            env_id=args.env,
            labels_file=args.labels_file,
            record_video=args.render,
            fps=args.fps)
        self.args.config['callbacks'] = self.callback

    def run(self, num_rollouts: int):
        # copied from ray.rllib.evaluate.run
        ray.init(local_mode=self.args.local_mode)

        # Create the Algorithm from config.
        cls = get_trainable_cls(self.args.run)
        algorithm = cls(config=self.args.config)

        # Load state from checkpoint, if provided.
        checkpoint = self.args.checkpoint
        if checkpoint:
            if os.path.isdir(checkpoint):
                checkpoint_dir = checkpoint
            else:
                checkpoint_dir = str(Path(checkpoint).parent)
            logging.info(f'Restoring algorithm from {checkpoint_dir}')
            restore_result = _TrainingResult(
                checkpoint=Checkpoint.from_directory(checkpoint_dir), metrics={}
            )
            algorithm.restore(restore_result)

        # Do the actual rollout.
        with RolloutSaver(
                outfile=self.args.out,
                use_shelve=self.args.use_shelve,
                write_update_file=self.args.track_progress,
                target_steps=0,  # to not constrain on num steps
                target_episodes=num_rollouts,
                save_info=self.args.save_info,
        ) as saver:
            rollout(algorithm, self.args.env, 0, num_rollouts, saver, not self.args.render)
        algorithm.stop()

    def collect_data(self, num_rollouts: int) -> Rollouts:

        # run rllib evaluation script with correct args and dummy arg parser
        self.run(num_rollouts=num_rollouts)

        # gets data back from callbacks by loading individual episodes results
        ep_files = get_files_with_extension(self.temp_dir, 'pkl.gz')
        rollouts = OrderedDict()
        logging.info(f'Gathering {len(ep_files)} episode rollouts...')
        for ep_file in tqdm.tqdm(ep_files):
            rollout: Rollout = load_object(ep_file)
            rollouts[rollout.rollout_id] = rollout  # organizes by rollout id

        return rollouts

    def remove_temp_data(self):
        # delete output temp dir
        shutil.rmtree(self.temp_dir)
