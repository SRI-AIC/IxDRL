import argparse
import logging
import tqdm
import pandas as pd
from typing import Dict
from ixdrl.util.io import get_file_name_without_extension

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

CLUSTER_ID_COL = 'Cluster'
ROLLOUT_ID_COL = 'Rollout'


def add_cluster_args(parser: argparse.ArgumentParser,
                     cluster_id_col: str = CLUSTER_ID_COL,
                     rollout_id_col: str = ROLLOUT_ID_COL,
                     required: bool = True):
    """
    Utility method that adds command-line parameters necessary to load clustering information about rollouts from a CSV
    file. The CSV file shall have one row per rollout, with one column specifying the rollout's ID and another
    specifying the cluster ID to which the rollout belongs to.
    Two parameters will be added to the given `argparse` parser which can be retrieved after parsing:
    - `"rollout_col"`: specifies rollout id column in the clusters CSV file.
    - `"cluster_col"`: specifies cluster id column in the clusters CSV file.
    :param argparse.ArgumentParser parser: the existing argument parser to add the clustering parameters.
    :param str cluster_id_col: the default name of the column in the clusters file holding the information on the
    cluster id for each rollout.
    :param rollout_id_col: the default name of the column in the clusters file holding the information on the id for
    each rollout.
    :param bool required: whether to require setting the clusters argument.
    """
    parser.add_argument('--clusters', '-cl', type=str, default=None, required=required,
                        help='The path to the clusters CSV file to be used, containing a reference for '
                             'the replay file of each trace and corresponding cluster. Data will be '
                             'processed for each cluster and saved to separate directories.')
    parser.add_argument('--cluster-col', type=str, default=cluster_id_col,
                        help='The name of the column in the clusters file holding the information on the cluster id '
                             'for each rollout.')
    parser.add_argument('--rollout-col', type=str, default=rollout_id_col,
                        help='The name of the column in the clusters file holding the information on the id for each '
                             'rollout.')


def organize_interestingness_by_cluster(interestingness_df: pd.DataFrame,
                                        clusters_df: pd.DataFrame,
                                        args: argparse.Namespace,
                                        use_tqdm: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Organizes a given interestingness dataframe according to the given rollout clustering data.
    :param pd.DataFrame interestingness_df: the interestingness dataframe, likely produced by the
    :ref:`ixdrl.bin.analyze` script.
    :param pd.DataFrame clusters_df: the dataframe containing the rollout clustering data, containing a row per rollout,
    with one column specifying the rollout's ID and another specifying the cluster ID to which the rollout belongs to.
    :param argparse.Namespace args: the `argparse` result, assuming the arguments were added to the parser using the
    :func:`add_cluster_args` function.
    :param bool use_tqdm: whether to use the `tqdm` module while processing each cluster.
    :rtype: dict[str, pd.DataFrame]
    :return: a dictionary in the form "cluster id": "interestingness dataframe", containing the interestingness data
    for each cluster. A new column named `args.cluster_col` is added to each cluster's dataframe with the cluster ID.
    """

    # get only file name fom replay to match rollout id
    clusters_df[args.rollout_col] = clusters_df[args.rollout_col].astype(str).apply(get_file_name_without_extension)
    rollout_ids = clusters_df[args.rollout_col].unique()
    missing = set(rollout_ids) - set(interestingness_df[ROLLOUT_ID_COL].unique())
    if len(missing) > 0:
        logging.warning(f'Could not find interestingness data for replays/rollouts:')
        logging.warning('\n'.join(missing))
    else:
        logging.info(f'Interestingness data found for all {len(rollout_ids)} rollouts')

    clusters_dfs = clusters_df.groupby(args.cluster_col)
    if use_tqdm:
        clusters_dfs = tqdm.tqdm(clusters_dfs, total=len(clusters_df[args.cluster_col].unique()))

    clusters_int_dfs: Dict[str, pd.DataFrame] = {}
    for cluster_id, cluster_df in clusters_dfs:
        cluster_rollout_ids = cluster_df[args.rollout_col]
        cluster_int_df = interestingness_df[interestingness_df[ROLLOUT_ID_COL].isin(cluster_rollout_ids)].copy()
        cluster_int_df.loc[:, args.cluster_col] = cluster_id  # add column with the cluster number
        clusters_int_dfs[cluster_id] = cluster_int_df

    return clusters_int_dfs
