import argparse
import logging
import os
import pandas as pd
import random
import shutil
import tqdm
from typing import Dict, List, Optional

from ixdrl import Rollouts
from ixdrl.bin import add_cluster_args
from ixdrl.util.cmd_line import str2bool, save_args
from ixdrl.util.io import create_clear_dir, load_object
from ixdrl.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__desc__ = 'Copies a number of videos (replays) as specified in a loaded RolloutData file. ' \
           'Possibly loads a cluster file and selects/copies videos for each cluster.'


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Interaction data file from which to extract video file names of rollouts.')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Directory in which to save videos.')
    parser.add_argument('--amount', '-a', type=int, default=None,
                        help='Number of videos to be selected and copied over to the output directory.'
                             'If not specified, videos for all rollouts will be copied.')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='Random seed to be used for selection of videos to be copied.'
                             'If not specified, videos will be selected sequentially.')

    parser.add_argument('--video-dir', type=str, default=None,
                        help='Path to the location videos, if not as specified in the interaction data file. '
                             'In this case, the path to the video will be replaced by this argument.')

    parser.add_argument('--clear', '-c', type=str2bool, help='Clear output directories before generating results.')
    parser.add_argument('--verbosity', '-v', type=int, default=0, help='Verbosity level.')

    add_cluster_args(parser)  # add arguments to load clusters CSV file
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise ValueError(f'Could not find interaction data file in {args.input}')

    create_clear_dir(args.output, args.clear)
    save_args(args, os.path.join(args.output, 'args.json'))
    change_log_handler(os.path.join(args.output, 'copy-videos.log'), args.verbosity)

    # load interaction data
    logging.info('========================================')
    logging.info(f'Loading interaction data from: {args.input}...')
    interaction_data: Rollouts = load_object(args.input)
    logging.info(f'Loaded data for {len(interaction_data)} rollouts')

    logging.info('========================================')
    videos_dest: Dict[str, List[str]] = {}  # stores videos to be copied and their destination directory

    # checks clusters mode
    clusters_df = None
    if args.clusters is not None and os.path.isfile(args.clusters):
        clusters_df = pd.read_csv(args.clusters, dtype={args.cluster_col: int, args.rollout_col: str})
        if args.rollout_col in clusters_df and args.cluster_col in clusters_df:
            num_clusters = len(clusters_df[args.cluster_col].unique())
            logging.info(f'Loaded cluster information for {len(clusters_df)} rollouts ({num_clusters} clusters) '
                         f'from {args.clusters}')

            logging.info('Selecting videos for each cluster...')
            for cluster, c_df in tqdm.tqdm(clusters_df.groupby(args.cluster_col), total=num_clusters):
                rollout_ids = list(c_df[args.rollout_col].unique())
                output_dir = os.path.join(args.output, str(cluster))
                videos_dest[output_dir] = _select_videos(interaction_data, rollout_ids,
                                                         args.amount, args.video_dir, args.seed)

    if clusters_df is None:
        # normal mode, select from all rollouts
        logging.info('Selecting videos for all rollouts')
        videos_dest[args.output] = _select_videos(interaction_data, None, args.amount, args.video_dir, args.seed)

    logging.info('========================================')
    num_videos = sum(len(v) for v in videos_dest.values())
    if num_videos == 0:
        logging.info('Could not find any videos from the given parameters.')
        return

    logging.info(f'Copying {num_videos} into {args.output}...')
    for output_dir, video_paths in videos_dest.items():
        if output_dir != args.output:
            create_clear_dir(output_dir, args.clear)
        for video_path in video_paths:
            logging.info(f'Copying {video_path} -> {output_dir}')
            shutil.copy(video_path, os.path.join(output_dir, os.path.basename(video_path)))

    logging.info('Done!')


def _select_videos(interaction_data: Rollouts, rollout_ids: Optional[List[str]],
                   amount: Optional[int], video_dir: Optional[str], seed: Optional[int]) -> List[str]:
    if rollout_ids is None:
        rollout_ids = list(interaction_data.keys())  # no filtering, use all rollouts
    if amount is None:
        amount = len(rollout_ids)  # select all rollouts
    if seed is not None:
        random.Random(seed).shuffle(rollout_ids)  # randomly shuffles rollouts for selection

    video_paths = []
    for rollout_id in rollout_ids:
        video_path = interaction_data[rollout_id].video_file
        if video_dir is not None:
            video_path = os.path.join(video_dir, os.path.basename(video_path))  # replace path to video
        if os.path.isfile(video_path):
            video_paths.append(video_path)  # add to selection to be copied
        if len(video_paths) >= amount:
            break  # don't select any more videos

    return video_paths


if __name__ == '__main__':
    main()
