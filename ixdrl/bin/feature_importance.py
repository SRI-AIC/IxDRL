import argparse
import logging
import os

import pandas as pd
import tqdm

from ixdrl.analysis import ROLLOUT_ID_COL
from ixdrl.bin import add_cluster_args
from ixdrl.interpretation.feature_importance import feature_importance_from_observations
from ixdrl.util.cmd_line import str2log_level, str2bool, save_args
from ixdrl.util.io import create_clear_dir
from ixdrl.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Loads a rollout clusters file, interaction and interestingness data and analyzes feature ' \
                  'importance, where relevance is taken from a model regressing interestingness dimensions\' values' \
                  'given the agent\'s observational features.'


def main():
    # create arg parser
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--interaction-data', '-d', type=str, required=True,
                        help='Pickle file containing the interaction data collected using `bin.collect.*` scripts.')
    parser.add_argument('--interestingness', '-i', type=str, default=None, required=True,
                        help='Path to pandas dataframe pickle file containing the interestingness data, produced '
                             'by the `bin.analyze` script.')
    parser.add_argument('--output', '-o', type=str, default=None, required=True,
                        help='The path to the directory in which to save the results.')
    parser.add_argument('--highlights', '-hi', type=str,
                        help='Path to CSV file containing the highlights/outliers info for which to produce '
                             'local/individual explanations.')

    parser.add_argument('--seed', type=int, default=17, help='Seed used for random number generation.')
    parser.add_argument('--processes', type=int, default=-1,
                        help='Number of processes for parallel processing. Value < 1 uses all available cpus.')
    parser.add_argument('--format', '-if', type=str, default='pdf', help='The format of image files.')
    parser.add_argument('--clear', '-c', type=str2bool, help='Clear output directories before generating results.')
    parser.add_argument('--verbosity', '-v', type=str2log_level, default=0, help='Verbosity level.')

    add_cluster_args(parser, required=False)  # add arguments to load clusters CSV file

    args = parser.parse_args()

    # checks all inputs
    if not os.path.isfile(args.interaction_data):
        raise ValueError(f'Interaction data file does not exist: {args.interaction_data}')
    if not os.path.isfile(args.interestingness):
        raise ValueError(f'Interestingness dataframe file does not exist: {args.interestingness}')

    # checks output dir and log file
    output_dir = args.output
    create_clear_dir(output_dir, args.clear)
    log_file = os.path.abspath(os.path.join(output_dir, 'feature-importance.log'))
    change_log_handler(log_file, append=os.path.isfile(log_file), level=args.verbosity)
    save_args(args, os.path.join(output_dir, 'args.json'))

    logging.info('=========================================')
    logging.info('Loading data...')

    # tries to load highlights file
    highlights_df = pd.read_csv(args.highlights, dtype={ROLLOUT_ID_COL: str}) \
        if os.path.isfile(args.highlights) else None
    if highlights_df is not None:
        logging.info(f'Loaded highlights information corresponding to a total of {len(highlights_df)} outliers '
                     f'from {args.highlights}')

    # performs feature importance analysis over whole dataset
    logging.info('Analyzing feature importance for whole dataset...')
    feature_importance_from_observations(
        args.interaction_data,
        args.interestingness,
        output_dir=output_dir,
        highlights_df=highlights_df,
        processes=args.processes,
        seed=args.seed,
        img_format=args.format
    )

    if args.clusters is not None and os.path.isfile(args.clusters):
        # load clusters file
        clusters_df: pd.DataFrame = pd.read_csv(args.clusters)
        num_clusters = len(clusters_df[args.cluster_col].unique())
        logging.info(f'Loaded clusters file from: {args.clusters} '
                     f'({num_clusters} clusters, {len(clusters_df)} rollouts).')

        logging.info(f'Analyzing feature importance for {num_clusters} clusters...')
        for cluster, cluster_df in tqdm.tqdm(clusters_df.groupby(args.cluster_col)):
            rollout_ids = list(cluster_df[ROLLOUT_ID_COL].unique())
            logging.info(f'Processing cluster "{cluster}" ({len(rollout_ids)} rollouts)...')
            feature_importance_from_observations(
                args.interaction_data,
                args.interestingness,
                output_dir=output_dir,
                highlights_df=highlights_df,
                rollout_ids=rollout_ids,
                processes=args.processes,
                seed=args.seed,
                img_format=args.format
            )


if __name__ == "__main__":
    main()
