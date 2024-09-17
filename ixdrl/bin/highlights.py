import argparse
import json
import logging
import os
import pandas as pd
from typing import Dict

from ixdrl.analysis import TIMESTEP_COL, ROLLOUT_ID_COL
from ixdrl.bin import add_cluster_args, organize_interestingness_by_cluster
from ixdrl.bin.analyze import INTERESTINGNESS_PLOTS_FILE, INTERESTINGNESS_PANDAS_FILE
from ixdrl.interpretation.highlights import extract_highlights
from ixdrl.util.cmd_line import str2log_level, str2bool, save_args
from ixdrl.util.io import create_clear_dir, load_object
from ixdrl.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Loads a trace clusters file and interestingness data and extracts highlights.'

METADATA_COLS = [TIMESTEP_COL, ROLLOUT_ID_COL]


def main():
    # create arg parser
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--output', '-o', type=str, default=None, required=True,
                        help='The path to the directory in which to save the results.')
    parser.add_argument('--interestingness', '-i', type=str, default=None, required=True,
                        help='The path to the directory with the interestingness analyses for all traces '
                             'and all the plots. Typically this was produced by the "analyze.py" script.')

    parser.add_argument('--metadata', '-m', type=str, default=None, required=True,
                        help='The path to the rollouts metadata file containing the names of the replay video files.')
    parser.add_argument('--max-highlights', '-mh', type=int, default=10,
                        help='Maximum highlights to be extracted per interestingness dimension.')
    parser.add_argument('--record-timesteps', '-t', type=int, default=41,
                        help='The number of environment time-steps to be recorded in each video.')
    parser.add_argument('--fade-ratio', '-f', type=float, default=0.25,
                        help='The ratio of frames to which apply a fade-in/out effect.')
    parser.add_argument('--iqr-mul', type=float, default=1.5,
                        help='The IQR multiplier to determine outliers.')

    parser.add_argument('--processes', type=int, default=-1,
                        help='Number of processes for parallel processing. Value < 1 uses all available cpus.')
    parser.add_argument('--format', '-if', type=str, default='pdf', help='The format of image files.')
    parser.add_argument('--clear', '-c', type=str2bool, help='Clear output directories before generating results.')
    parser.add_argument('--verbosity', '-v', type=str2log_level, default=0, help='Verbosity level.')

    add_cluster_args(parser, required=False)  # add arguments to load clusters CSV file (optionally)

    args = parser.parse_args()

    # checks all inputs
    if not os.path.isdir(args.interestingness):
        raise ValueError(f'Interestingness directory does not exist: {args.interestingness}')
    int_pandas_file = os.path.join(args.interestingness, INTERESTINGNESS_PANDAS_FILE)
    if not os.path.isfile(int_pandas_file):
        raise ValueError(f'Interestingness dataset file does not exist: {int_pandas_file}')
    int_plots_file = os.path.join(args.interestingness, INTERESTINGNESS_PLOTS_FILE)
    if not os.path.isfile(int_plots_file):
        raise ValueError(f'Interestingness plots file does not exist: {int_plots_file}')
    if not os.path.isfile(args.metadata):
        raise ValueError(f'Rollouts metadata file does not exist: {args.metadata}')

    # checks output dir and log file
    output_dir = args.output
    create_clear_dir(output_dir, args.clear)
    log_file = os.path.abspath(os.path.join(output_dir, 'highlights.log'))
    change_log_handler(log_file, append=os.path.isfile(log_file), level=args.verbosity)
    save_args(args, os.path.join(output_dir, 'args.json'))

    logging.info('=========================================')

    # load interestingness files
    interestingness_df: pd.DataFrame = pd.read_pickle(int_pandas_file)
    interestingness_df.reset_index(drop=True, inplace=True)  # resets index in case it's timestep indexed
    logging.info(f'Loaded interestingness dataset from: {int_pandas_file} '
                 f'({len(interestingness_df.columns[2:])} dimensions, '
                 f'{len(interestingness_df[ROLLOUT_ID_COL].unique())} rollouts).')
    interestingness_plots: Dict[str, Dict[str, str]] = load_object(int_plots_file)
    logging.info(f'Loaded interestingness plots from: {int_plots_file} '
                 f'({len(next(iter(interestingness_plots.values())))} dimensions, '
                 f'{len(interestingness_plots)} rollouts).')

    # load interaction metadata file with paths to replay video files
    with open(args.metadata, 'r') as fp:
        rollout_metadata: Dict = json.load(fp)
    logging.info(f'Loaded interaction metadata from: {args.metadata} '
                 f'({len(rollout_metadata["rollouts"])} rollouts).')
    videos_dir = os.path.dirname(args.metadata)

    clusters_int_dfs: Dict[str, pd.DataFrame] = {}
    if args.clusters is not None and os.path.isfile(args.clusters):
        # load clusters file
        clusters_df = pd.read_csv(args.clusters)
        logging.info(f'Loaded clusters file from: {args.clusters} '
                     f'({len(clusters_df[args.cluster_col].unique())} clusters, {len(clusters_df)} rollouts).')

        logging.info('=========================================')
        logging.info('Organizing interestingness per cluster...')
        clusters_int_dfs = organize_interestingness_by_cluster(interestingness_df, clusters_df, args, use_tqdm=True)
    else:
        logging.info(f'Could not load clusters file from: {args.clusters}, '
                     f'highlights will be extracted for whole dataset')

    clusters_int_dfs['overall'] = interestingness_df  # add overall dataset

    # gets the name of the main dimensions
    dimensions = [d for d in interestingness_df.columns if d not in METADATA_COLS]
    for cluster_id, cluster_int_df in clusters_int_dfs.items():
        logging.info('=========================================')
        logging.info(f'Processing cluster {cluster_id} ({len(cluster_int_df[ROLLOUT_ID_COL].unique())} rollouts)...')
        output_path = os.path.join(output_dir, f'cluster-{cluster_id}')
        create_clear_dir(output_path, args.clear)

        logging.info('_____________________________________')
        logging.info('Extracting highlights...')
        extract_highlights(cluster_int_df, dimensions, output_path, interestingness_plots, rollout_metadata,
                           args.max_highlights, args.iqr_mul, args.record_timesteps, args.fade_ratio,
                           args.processes, args.format, videos_dir)

    logging.info('Done!')


if __name__ == "__main__":
    main()
