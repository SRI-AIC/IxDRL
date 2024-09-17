import logging
import os.path
from collections import OrderedDict
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import tqdm

from ixdrl import Rollouts
from ixdrl.analysis import ROLLOUT_ID_COL, TIMESTEP_COL
from ixdrl.analysis.goal_conduciveness import GoalConducivenessAnalysis
from ixdrl.analysis.incongruity import IncongruityAnalysis
from ixdrl.interpretation.explanation import explain_models_global, explain_highlights
from ixdrl.interpretation.feature_extraction import extract_features_from_rollout
from ixdrl.interpretation.regression import train_models, evaluate_models, load_models, \
    select_train_test_split
from ixdrl.util.io import create_clear_dir, load_object, get_file_changed_extension
from ixdrl.util.mp import run_parallel

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

TARGETS_FILE = 'targets-df.pkl.gz'
FEATURES_FILE = 'features-df.pkl.gz'
META_COLUMNS = [ROLLOUT_ID_COL, TIMESTEP_COL]

MAX_NUM_FEATURES = 20  # number of features for importance analysis
MAX_HIGHLIGHTS = 10
TEST_FRACTION = 0.2  # ratio of data used for model testing
EXPLAIN_FRACTION = 0.1  # ratio of test data used for global explanations


def feature_importance_from_observations(
        interaction_data_file: str,
        interestingness_file: str,
        output_dir: str,
        dimensions: Optional[List[str]] = None,
        highlights_df: Optional[pd.DataFrame] = None,
        rollout_ids: Optional[List[str]] = None,
        processes: int = -1,
        seed: Optional[int] = None,
        img_format: str = 'pdf'
):
    """
    Performs feature importance analysis for interestingness by learning a regression model from agent observations
    (features) to interestingness variables (targets). The idea is to explain how the agent's input features affect
    interestingness at a global level, i.e., given the whole set of interaction data. First, it extracts high-level
    features from the observations in the interaction data according to the gym space specification. Then, it matches
    the feature values for each timestep of each rollout to the corresponding interestingness values and trains a
    Gradient Boosting regressor. It then uses SHAP values to identify important features, explain how the value of
    features affects each of the interestingness dimensions, identify feature dependencies and analyze the interactions.
    Note: if the features and interestingness dataset and the regression models already exist (files in the output
    directory), they will be loaded and only feature importance analysis will be performed.
    :param str interaction_data_file: the path to the file with interaction data, containing the observations for each
    timestep of each rollout.
    :param str interestingness_file: the path to the file with the interestingness dataframe, containing the value of
    each interestingness dimension/variable (columns) computed for each timestep of each rollout (rows).
    :param str output_dir: the path to the directory in which to save the regression models and various results.
    :param list[str] dimensions: the interestingness dimensions whose relationship to the features we want to analyze.
    A regression model will be created for each dimension. If `None`, all dimensions will be considered.
    :param pd.DataFrame highlights_df: the dataframe containing the highlights/outliers info for local explanations.
    :param list[str] rollout_ids: the list of rollout identifiers from which to compute the regression models, and/or
    from which to perform feature importance analysis. `None` will use all rollouts for which there is interaction and
    interestingness data.
    :param int processes: the number of parallel processes to use. Uses `joblib` convention.
    :param int seed: the seed for random number generation (for reproducibility).
    :param str img_format: the image format of plots that are generated.
    :return:
    """
    # create sub-dir to store data
    _output_dir = os.path.join(output_dir, 'data')
    create_clear_dir(_output_dir, clear=False)
    features_file = os.path.join(_output_dir, FEATURES_FILE)
    targets_file = os.path.join(_output_dir, TARGETS_FILE)

    if os.path.isfile(features_file) and os.path.isfile(targets_file):
        logging.info('=========================================')
        logging.info(f'Loading feature and target data from:\n\t{features_file}\n\t{targets_file}\n...')
        x = pd.read_pickle(features_file)
        y = pd.read_pickle(targets_file)
        logging.info(f'Loaded: {x.shape[0]} instances, {x.shape[1]} features, {y.shape[1]} targets')

    else:
        x, y = get_regression_data_from_observations(
            interaction_data_file, interestingness_file, dimensions,
            rollout_ids, features_file, targets_file, processes
        )

        if x is None or y is None:
            return  # could not extract data for regression

    analyze_feature_importance(x, y, output_dir, highlights_df, processes, seed, img_format)


def get_regression_data_from_observations(
        interaction_data_file: str,
        interestingness_file: str,
        dimensions: Optional[List[str]],
        rollout_ids: Optional[List[str]],
        features_file: str,
        targets_file: str,
        processes: int = -1) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Gets two dataframes to be used for training a regression model from observation features to interestingness
    dimensions. First, extracts high-level features from the observations in the interaction data according to the gym
    space specification. Then, it matches the feature values for each timestep of each rollout to the corresponding
    interestingness values.
    :param str interaction_data_file: the path to the file with interaction data, containing the observations for each
    timestep of each rollout.
    :param str interestingness_file: the path to the file with the interestingness dataframe, containing the value of
    each interestingness dimension/variable (columns) computed for each timestep of each rollout (rows).
    :param list[str] dimensions: the interestingness dimensions whose relationship to the features we want to analyze.
    A regression model will be created for each dimension.
    :param list[str] rollout_ids: the list of rollout identifiers from which to compute the regression models, and/or
    from which to perform feature importance analysis. `None` will use all rollouts for which there is interaction and
    interestingness data.
    :param str features_file: the path to the file in which to save the features (observations) dataframe.
    :param str targets_file: the path to the file in which to save the targets (interestingness) dataframe.
    :param int processes: the number of parallel processes to use. Uses `joblib` convention.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    :return: a tuple containing: the features (observations) dataframe; the targets (interestingness) dataframe.
    """
    logging.info('=========================================')
    logging.info(f'Loading interestingness dataset from: {interestingness_file}...')
    interestingness_df: pd.DataFrame = pd.read_pickle(interestingness_file)
    interestingness_df.reset_index(drop=True, inplace=True)  # resets index in case it's timestep indexed
    logging.info(f'Loaded data for {len(interestingness_df.columns[2:])} dimensions, '
                 f'{len(interestingness_df[ROLLOUT_ID_COL].unique())} rollouts')

    logging.info('_________________________________________')
    logging.info(f'Loading interaction dataset from: {interaction_data_file}...')
    interaction_data: Rollouts = load_object(interaction_data_file)
    logging.info(f'Loaded interaction data for {len(interaction_data)} rollouts')

    logging.info('_________________________________________')
    logging.info('Checking rollouts...')

    # check data consistency for rollout ids
    data_ids = set(interaction_data.keys())
    int_ids = set(interestingness_df[ROLLOUT_ID_COL].unique())
    rollout_ids = data_ids if rollout_ids is None else set(rollout_ids)  # assume all rollouts in the data
    no_data_ids = rollout_ids - (data_ids.intersection(int_ids))  # takes difference
    if len(no_data_ids) == 0:
        logging.info(f'Found interaction and interestingness data for all {len(rollout_ids)} rollouts')
    elif len(no_data_ids) == len(rollout_ids):
        logging.error(f'Could not find interaction and/or interestingness data for any of the '
                      f'{len(rollout_ids)} rollouts!')
        return None, None
    else:
        logging.info(f'Could not find interaction and/or interestingness data for {len(no_data_ids)} rollouts: '
                     f'{no_data_ids}')

    rollout_ids = sorted(rollout_ids - no_data_ids)  # selects rollouts for which there are data

    logging.info('_________________________________________')
    logging.info('Transforming observations to numerical features...')
    args = [interaction_data[rollout_id] for rollout_id in rollout_ids]
    features_dfs: List[pd.DataFrame] = run_parallel(extract_features_from_rollout, args, processes, use_tqdm=True)
    features_df = pd.concat(features_dfs, axis=0)  # vertically concatenate all rollouts' dataframes
    logging.info(f'Extracted {len(features_df.columns) - 2} features for a total of {len(features_df)} timesteps')

    logging.info('_________________________________________')
    logging.info('Merging and aligning features and interestingness data...')
    interestingness_df = interestingness_df[interestingness_df[ROLLOUT_ID_COL].isin(rollout_ids)]
    df = pd.merge(features_df, interestingness_df, how='left', on=META_COLUMNS)

    logging.info('_________________________________________')
    logging.info('Preparing data...')
    features_cols = list(features_df.columns[2:])  # ignore rollout id, timestep

    x = df[META_COLUMNS + features_cols]
    logging.info(f'Saving dataset for {len(x.columns) - 2} features to {features_file}...')
    x.to_pickle(features_file)
    x.describe(include='all').to_csv(get_file_changed_extension(features_file, 'csv', suffix='-stats'))

    if dimensions is None:
        y = df[META_COLUMNS + [col for col in interestingness_df.columns if col not in META_COLUMNS]]
    else:
        y = df[META_COLUMNS + [col for col in dimensions if col in df.columns]]
    logging.info(f'Saving dataset for {len(y.columns) - 2} targets to {targets_file}...')
    y.to_pickle(targets_file)
    y.describe(include='all').to_csv(get_file_changed_extension(targets_file, 'csv', suffix='-stats'))

    return x, y


def analyze_feature_importance(x: pd.DataFrame,
                               y: pd.DataFrame,
                               output_dir: str,
                               highlights_df: Optional[pd.DataFrame],
                               processes: int = -1,
                               seed: int = 17,
                               img_format: str = 'pdf'):
    """
    Performs feature importance analysis for interestingness by learning a regression model from features to some
    target variables.
    :param pd.DataFrame x: the dataset containing the features (columns) for each data instance (rows).
    :param pd.DataFrame y: the dataset containing the targets (columns) for each data instance (rows). The `x` and `y`
    datasets need to be aligned.
    :param str output_dir: the path to the directory in which to save the regression models and various results.
    :param pd.DataFrame highlights_df: the dataframe containing the highlights/outliers info for local explanations.
    :param int processes: the number of parallel processes to use. Uses `joblib` convention.
    :param int seed: the seed for random number generation (for reproducibility).
    :param str img_format: the image format of plots that are generated.
    :return:
    """
    # select columns
    features_cols = [col for col in x.columns if col not in META_COLUMNS]  # features / inputs without meta columns
    target_cols = [col for col in y.columns if col not in META_COLUMNS]  # targets / outputs without meta columns

    logging.info('=========================================')
    logging.info(f'Training ensemble of {len(target_cols)} regression models: features -> targets...')

    logging.info('_________________________________________')
    # tries to load models
    _output_dir = os.path.join(output_dir, 'models')
    create_clear_dir(_output_dir, clear=False)
    model_paths = [os.path.join(_output_dir, f'model-{target_cols[i].lower()}.ubj')
                   for i in range(len(target_cols))]
    models = load_models(model_paths, target_cols)
    needs_train_idxs = [i for i in range(len(models)) if models[i] is None]

    logging.info('_________________________________________')
    logging.info(f'Preparing data for {len(target_cols)} dimensions...')
    x_sets: List[pd.DataFrame] = []
    y_sets: List[pd.DataFrame] = []
    for i in tqdm.tqdm(range(len(target_cols))):
        # either loads the data specific for this dimension or select from original data
        dim = target_cols[i]
        features_file = os.path.join(output_dir, 'data', f'{dim.lower()}-{FEATURES_FILE}')
        targets_file = os.path.join(output_dir, 'data', f'{dim.lower()}-{TARGETS_FILE}')
        if os.path.isfile(features_file) and os.path.isfile(targets_file):
            _x = pd.read_pickle(features_file)
            _y = pd.read_pickle(targets_file)
        else:
            _x, _y = _get_data_for_dimension(x, features_cols, y, dim)
            _x.to_pickle(features_file)
            _y.to_pickle(targets_file)

        x_sets.append(_x)
        y_sets.append(_y)

    x_train, y_train, x_test, y_test = select_train_test_split(
        x_sets, y_sets, test_fraction=TEST_FRACTION, labels=target_cols, seed=seed)

    # trains models whose file could not be found
    if len(needs_train_idxs) > 0:
        _models = train_models(needs_train_idxs, x_train, y_train, x_test, y_test, model_paths,
                               labels=target_cols, processes=processes, seed=seed)
        for i, idx in enumerate(needs_train_idxs):
            models[idx] = _models[i]  # set the returned model that has the trained weights

        # evaluate models
        logging.info('=========================================')
        evaluate_models(models, model_paths, x_test, y_test, output_dir,
                        max_num_features=MAX_NUM_FEATURES, labels=target_cols, img_format=img_format)

    logging.info('=========================================')
    logging.info('Explaining prediction models overall...')
    _output_dir = os.path.join(output_dir, 'global-feature-explanations')
    create_clear_dir(_output_dir, clear=False)

    # select subset of test for global explanations
    _, _, x_explain, _ = select_train_test_split(
        x_test, y_test, test_fraction=EXPLAIN_FRACTION, labels=target_cols, seed=seed)

    # get stats for feature data used for interpretation
    for i, _x in enumerate(x_explain):
        label = target_cols[i]
        _x.describe(include='all').to_csv(os.path.join(_output_dir, f'test-feat-stats-{label.lower()}.csv'))
        logging.info(f'Selected {len(_x)} random feature samples from the data for explanations of {label}')

    explainers = explain_models_global(
        models, x_explain, MAX_NUM_FEATURES, _output_dir, img_format, target_cols, processes)
    logging.info(f'Saved global explanation results to {_output_dir}')

    if highlights_df is not None and len(highlights_df) > 0:
        _x_sets = [pd.merge(x.loc[:, [ROLLOUT_ID_COL, TIMESTEP_COL]], x_sets[i], left_index=True, right_index=True)
                   for i in range(len(x_sets))] # add rollout and timestep cols to allow matching highlights
        logging.info('=========================================')
        logging.info('Explaining individual predictions...')
        _output_dir = os.path.join(output_dir, 'local-feature-explanations')
        create_clear_dir(_output_dir, clear=False)
        explain_highlights(explainers, _x_sets, highlights_df, MAX_HIGHLIGHTS, int(MAX_NUM_FEATURES / 2),
                           _output_dir, target_cols, img_format, 1)  # TODO processes)
        logging.info(f'Saved local explanation results to {_output_dir}')
    else:
        logging.warning('No highlights information provided, so no local explanations will be generated')


def _get_data_for_dimension(x: pd.DataFrame, features_cols: List[str],
                            y: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = y.loc[:, target_col].to_frame()  # select target column values

    # checks for dimensions that use data from multiple timesteps
    gc_name = GoalConducivenessAnalysis().name
    if target_col == gc_name:
        n_steps = np.where(~np.isnan(y))[0][0]  # estimate num. timesteps needed to compute Goal Cond. from data
        return _get_temporal_features(x, y, features_cols, n_steps)  # return data with temporal-based features
    inc_name = IncongruityAnalysis().name
    if target_col == inc_name:
        return _get_temporal_features(x, y, features_cols, 1)  # incongruity requires 2 timesteps

    x = x.loc[:, features_cols]  # keep only feature columns
    return x, y  # otherwise simply return the given datasets


def _get_temporal_features(x: pd.DataFrame, y: pd.DataFrame,
                           features_cols: List[str], n_steps: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # time window has length n_steps + 1
    x_dfs = []

    for _, x_df in x.groupby(ROLLOUT_ID_COL):
        # get transformed features for this episode; n_steps won't have data
        data: Dict[str, np.ndarray] = OrderedDict({})
        for f in features_cols:
            # get transformed features from values of each feature
            _x = x_df.loc[:, f].values
            # data[f'{f} diff'] = _x[n_steps:] - _x[:-n_steps]  # add diff between last and first step in time window
            # data[f'{f} mean'] = np.convolve(_x, np.ones(n_steps + 1), 'valid') / (n_steps + 1)  # add time window mean
            # data[f'{f}'] = _x[:-n_steps]  # add original features
            for i in range(n_steps + 1):  # add all timesteps (concatenate)
                data[f'{f} t-{i}'] = _x[n_steps - i:len(_x) - i]
            # if n_steps % 2 == 0:
            #     data[f'{f} fd'] = finite_diff(_x, order=1, accuracy=n_steps, fd_type='backward')[n_steps:]
            #     data[f'{f} nd'] = normalized_derivative(_x, order=1, accuracy=n_steps, fd_type='backward')[n_steps:]
            # else:
            #     data[f'{f} diff'] = _x[n_steps:] - _x[:-n_steps]  # add diff between last and first step in time window
            # TODO add other features, eg mean value, std? append original features at t? finite diffs?

        x_dfs.append(pd.DataFrame(data, index=x_df.iloc[n_steps:].index))  # set original index to match y later

    x = pd.concat(x_dfs, axis=0)  # append all episode dfs with transformed features
    y = y.loc[x.index]  # select from target to match features
    return x, y
