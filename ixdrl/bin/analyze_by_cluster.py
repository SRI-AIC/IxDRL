import argparse
import logging
import os
import pandas as pd
import plotly.express as px
import tqdm
from typing import Dict, List

from ixdrl.analysis import print_stats, TIMESTEP_COL
from ixdrl.analysis.full import FullAnalysis
from ixdrl.bin import add_cluster_args, ROLLOUT_ID_COL
from ixdrl.bin.analyze import INTERESTINGNESS_PLOTS_FILE, INTERESTINGNESS_PANDAS_FILE
from ixdrl.util.cmd_line import str2log_level, str2bool, save_args
from ixdrl.util.io import create_clear_dir, get_file_name_without_extension, save_object, load_object
from ixdrl.util.logging import change_log_handler
from ixdrl.util.plot import plot_bar, dummy_plotly, plot_radar

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Loads a trace clusters file and interestingness data and performs analyses per cluster,' \
                  'plotting several comparative charts.'

DISC_PALETTE_1 = px.colors.qualitative.Bold  # T10  # Pastel
DISC_PALETTE_2 = px.colors.qualitative.Light24
DISC_PALETTE = px.colors.diverging.Portland


def main():
    # create arg parser
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--output', '-o', type=str, default=None, required=True,
                        help='The path to the directory in which to save the collected interaction data.')
    parser.add_argument('--interestingness', '-i', type=str, default=None, required=True,
                        help='The path to the directory with the interestingness analyses for all traces '
                             'and all the plots. Typically this was produced by the "analyze.py" script.')

    parser.add_argument('--format', '-if', type=str, default='pdf', help='The format of image files.')
    parser.add_argument('--clear', '-c', type=str2bool, help='Clear output directories before generating results.')
    parser.add_argument('--verbosity', '-v', type=str2log_level, default=0, help='Verbosity level.')

    add_cluster_args(parser)  # add arguments to load clusters CSV file
    args = parser.parse_args()

    # checks all inputs
    if not os.path.isfile(args.clusters):
        raise ValueError(f'Clusters file does not exist: {args.clusters}')
    if not os.path.isdir(args.interestingness):
        raise ValueError(f'Interestingness directory does not exist: {args.interestingness}')
    int_pandas_file = os.path.join(args.interestingness, INTERESTINGNESS_PANDAS_FILE)
    if not os.path.isfile(int_pandas_file):
        raise ValueError(f'Interestingness dataset file does not exist: {int_pandas_file}')
    int_plots_file = os.path.join(args.interestingness, INTERESTINGNESS_PLOTS_FILE)
    if not os.path.isfile(int_plots_file):
        raise ValueError(f'Interestingness plots file does not exist: {int_plots_file}')

    # checks output dir and log file
    output_dir = args.output
    create_clear_dir(output_dir, args.clear)
    log_file = os.path.abspath(os.path.join(output_dir, 'interestingness-by-clusters.log'))
    change_log_handler(log_file, append=os.path.isfile(log_file), level=args.verbosity)
    save_args(args, os.path.join(output_dir, 'args.json'))

    logging.info('=========================================')

    # load clusters file
    clusters_df = pd.read_csv(args.clusters, dtype={args.cluster_col: int, args.rollout_col: str})
    logging.info(f'Loaded clusters file from: {args.clusters} '
                 f'({len(clusters_df[args.cluster_col].unique())} clusters, {len(clusters_df)} replays).')

    # load interestingness files
    interestingness_df: pd.DataFrame = pd.read_pickle(int_pandas_file)
    interestingness_df.reset_index(drop=True, inplace=True)  # resets index in case it's timestep indexed
    logging.info(f'Loaded interestingness dataset from: {int_pandas_file} '
                 f'({len(interestingness_df.columns[2:])} dimensions, '
                 f'{len(interestingness_df[ROLLOUT_ID_COL].unique())} replays).')
    interestingness_plots: Dict[str, Dict[str, str]] = load_object(int_plots_file)
    logging.info(f'Loaded interestingness plots from: {int_plots_file} '
                 f'({len(next(iter(interestingness_plots.values())))} dimensions, '
                 f'{len(interestingness_plots)} replays).')

    logging.info('=========================================')
    logging.info('Verifying replay/rollout data...')
    # get only file name fom replay to match rollout id
    clusters_df[args.rollout_col] = clusters_df[args.rollout_col].apply(get_file_name_without_extension)
    missing = set(clusters_df[args.rollout_col].unique()) - set(interestingness_df[ROLLOUT_ID_COL].unique())
    if len(missing) > 0:
        logging.warning(f'Could not find interestingness data for replays/rollouts:')
        logging.warning('\n'.join(missing))
    else:
        logging.info(f'Interestingness data found for all {len(clusters_df[args.rollout_col].unique())} replays')

    logging.info('=========================================')
    logging.info('Organizing interestingness per cluster...')
    clusters_int_dfs: Dict[int, pd.DataFrame] = {}
    for cluster_id, cluster_df in tqdm.tqdm(clusters_df.groupby(args.cluster_col),
                                            total=len(clusters_df[args.cluster_col].unique())):
        cluster_int_df = interestingness_df[
            interestingness_df[ROLLOUT_ID_COL].isin(cluster_df[args.rollout_col])].copy()
        cluster_int_df[args.cluster_col] = cluster_id  # add column with the cluster number
        clusters_int_dfs[cluster_id] = cluster_int_df

    # gets the name of the main dimensions
    dimensions = [ia.name for ia in FullAnalysis({'dummy': None}).analyses if ia.name in interestingness_df]
    dummy_plotly()  # just to clear imports
    for cluster_id, cluster_int_df in clusters_int_dfs.items():
        logging.info('=========================================')
        logging.info(f'Processing cluster {cluster_id} ({len(cluster_int_df[ROLLOUT_ID_COL].unique())} rollouts)...')
        output_path = os.path.join(output_dir, f'cluster-{cluster_id}')
        create_clear_dir(output_path, args.clear)

        logging.info('_____________________________________')
        logging.info('Printing interestingness plots...')
        figures = print_stats(cluster_int_df.drop(args.cluster_col, axis=1), output_path, dimensions, args.format,
                              rollout_plots=False)
        figs_file = os.path.join(output_path, INTERESTINGNESS_PLOTS_FILE)
        save_object(figures, figs_file, compress_gzip=True)
        logging.info(f'Saved all plotly figures to "{figs_file}"')

    logging.info('=========================================')
    logging.info('Comparing interestingness across clusters...')
    output_path = os.path.join(output_dir, 'comparison')
    create_clear_dir(output_path, args.clear)
    comp_figs = {}

    logging.info('_____________________________________')
    logging.info('Comparing mean interestingness across clusters...')

    # chooses palette based on num clusters
    palette = DISC_PALETTE_1 if len(clusters_int_dfs) <= len(DISC_PALETTE_1) else \
        DISC_PALETTE_2 if len(clusters_int_dfs) <= len(DISC_PALETTE_2) else DISC_PALETTE

    # merge all clusters dfs and select only main dimensions and cluster id col
    clusters_int_df = pd.concat(clusters_int_dfs.values())[dimensions + [args.cluster_col]]
    fig = plot_radar(clusters_int_df, 'Mean Interestingness',
                     os.path.join(output_path, f'mean-interestingness.{args.format}'),
                     group_by=args.cluster_col, var_label='Dimension', value_label='Value',
                     # min_val=-1, max_val=1, plot_mean=True,
                     show_legend=True, palette=palette, alpha=0.1)
    comp_figs[fig.layout.title.text] = fig

    logging.info('_____________________________________')
    logging.info('Comparing mean interestingness across clusters for each dimension...')
    all_dims: List[str] = [dim for dim in interestingness_df.columns if dim not in [TIMESTEP_COL, ROLLOUT_ID_COL]]
    for dimension in tqdm.tqdm(all_dims):
        dim_df = pd.DataFrame([pd.Series(cluster_int_df[dimension].values, name=str(cluster_id))
                               for cluster_id, cluster_int_df in clusters_int_dfs.items()]).T
        fig = plot_bar(dim_df, f'Mean {dimension.title()}',
                       os.path.join(output_path, f'mean-{dimension.lower()}.{args.format}'),
                       x_label='Cluster', y_label='Value', palette=palette)
        comp_figs[fig.layout.title.text] = fig

    logging.info('_____________________________________')
    figs_file = os.path.join(output_path, 'comparison_plots.pkl.gz')
    save_object(comp_figs, figs_file, compress_gzip=True)
    logging.info(f'Saved all comparison plotly figures to "{figs_file}"')


if __name__ == "__main__":
    main()
