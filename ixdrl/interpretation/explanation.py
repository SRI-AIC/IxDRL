import logging
import os
from typing import List, Optional

import gymnasium as gym
import joblib
import numpy as np
import pandas as pd
import shap
import xgboost
from sklearn.preprocessing import StandardScaler

from ixdrl.interpretation import get_clean_filename, DIMS_PALETTE, FEATURES_LABEL_FONT_SIZE, shap_util
from ixdrl.interpretation.feature_extraction import extract_features
from ixdrl.interpretation.highlights import DIMENSION_COL, LABEL_COL, ROLLOUT_ID_COL, TIMESTEP_COL, VALUE_COL
from ixdrl.util.io import create_clear_dir, get_file_changed_extension
from ixdrl.util.mp import run_parallel
from ixdrl.util.plot import TITLE_FONT_SIZE, AXES_TITLE_FONT_SIZE, DEF_TEMPLATE, plot_bar, dummy_plotly, \
    plot_matrix

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

EXPECTED_VALUE_COL = 'Expected Value'
META_COLUMNS = [ROLLOUT_ID_COL, TIMESTEP_COL]

SHAP_BATCH_SIZE = 500  # size of features' batches to compute SHAP interaction values in parallel
MATRIX_PALETTE = 'Agsunset'  # palette for interaction effects matrix


def explain_highlights(explainers: List[shap.TreeExplainer],
                       x_sets: List[pd.DataFrame],
                       highlights_df: pd.DataFrame,
                       max_highlights: int,
                       max_num_features: int,
                       output_dir: str,
                       labels: List[str],
                       img_format: str = 'pdf',
                       processes: Optional[int] = -1):
    """
    Generates local (instance-level) explanations according to the given highlights (outliers) information. Namely,
    uses the `shap` library to create decision plots explaining the impact of each feature for the predicted value
    for each highlight instance.
    :param list[shap.TreeExplainer] explainers: the list of shap regression model explainers.
    :param list[pd.DataFrame] x_sets: the original features dataframes for each model, containing the features (columns)
    for each instance in the data (rows). Must include the rollout id and timestep metadata columns to match with the
    highlights instances.
    :param pd.DataFrame highlights_df: the dataframe containing the highlights/outliers info for local explanations.
    :param int max_highlights: the maximum number of highlights' decisions to explain for each dimension and label.
    :param int max_num_features: the maximum number of features to use in the individual explanations.
    :param str output_dir: the directory in which to save results.
    :param str img_format: the image format of the plots to be generated.
    :param list[str] labels: the labels for each model, must match the dimension names.
    :param int processes: the number of processes used in parallel. Uses `joblib` convention.
    """
    assert len(explainers) == len(labels), (f'Number of explainers ({len(explainers)}) != '
                                            f'number of dimension labels ({len(labels)})')

    # get the data instances corresponding to the highlights
    logging.info(f'Selecting feature data from {len(highlights_df)} highlights...')
    args = []
    for dim, group_df in highlights_df.groupby(DIMENSION_COL):
        if dim not in labels:
            logging.warning(f'Could not find model for dimension: {dim}, skipping...')
            continue
        lbl_idx = labels.index(dim)
        x = x_sets[lbl_idx]
        features_cols = [col for col in x.columns if col not in META_COLUMNS]  # features / inputs without meta columns
        x_highlights = []
        for label, group_df in group_df.groupby(LABEL_COL):
            for _, s in group_df.iloc[:min(len(group_df), max_highlights), :].iterrows():
                x_match = x[(x[ROLLOUT_ID_COL] == str(s[ROLLOUT_ID_COL])) & (x[TIMESTEP_COL] == s[TIMESTEP_COL])].copy()
                if len(x_match) > 0:
                    # replace dimension, label with rollout, timestep columns
                    x_match[DIMENSION_COL] = dim
                    x_match[LABEL_COL] = label
                    x_highlights.append(x_match[[DIMENSION_COL, LABEL_COL, ROLLOUT_ID_COL, TIMESTEP_COL] + features_cols])

        if len(x_highlights) == 0:
            logging.warning(f'Could not find highlights for dimension: {dim}, skipping...')
            continue

        x_highlights = pd.concat(x_highlights, axis=0, ignore_index=True)  # ignore index in original dataset
        logging.info(f'Got "{dim}" data for a total of {len(x_highlights)} highlights')
        x_highlights.to_csv(os.path.join(output_dir, f'{dim.lower()}-highlights-features.csv'), index=False)

        args.append((explainers[lbl_idx], x_highlights, features_cols,
                     max_num_features, output_dir, img_format, processes))

    # plot highlights for each target model
    logging.info(f'Plotting local explanations for {len(labels)} models based on {len(highlights_df)} highlights...')
    run_parallel(explain_highlights_for_model, args, processes, use_tqdm=True)


def explain_highlights_for_model(explainer: shap.TreeExplainer,
                                 x: pd.DataFrame,
                                 feature_cols: List[str],
                                 max_num_features: int,
                                 output_dir: str,
                                 img_format: str = 'pdf',
                                 processes: int = -1):
    """
    Generates local (instance-level) explanations according to the given highlights (outliers) information. Namely,
    uses the `shap` library to create decision plots explaining the impact of each feature for the predicted value
    for each highlight instance.
    :param shap.TreeExplainer explainer: the shap regression model explainer for a particular target.
    :param pd.DataFrame x: the features dataframe, containing the features (columns) for each instance (rows) to be
    explained. Must include the metadata columns with information about the corresponding highlights.
    :param list[str] feature_cols: the names of the feature columns.
    :param int max_num_features: the maximum number of features to use in the individual explanations.
    :param str output_dir: the directory in which to save results.
    :param str img_format: the image format of the plots to be generated.
    :param int processes: the number of processes used in parallel. Uses `joblib` convention.
    """
    if len(x) == 0:
        return  # no highlights to be processed

    # explain the decision for each highlight
    shap_values = explainer.shap_values(x.loc[:, feature_cols])
    for i in range(len(x)):
        xi = x.iloc[i]
        dim = xi[DIMENSION_COL]
        title = f'{xi[LABEL_COL].title()} {dim.title()}'
        file_suffix = f'{dim}-{xi[LABEL_COL]}-{xi[ROLLOUT_ID_COL]}.{img_format}'
        _plot_decision(shap_values[i], explainer.expected_value, xi[feature_cols], dim,
                       title, file_suffix, max_num_features, output_dir)


def explain_models_global(models: List[xgboost.XGBRegressor],
                          x: List[pd.DataFrame],
                          max_num_features: int,
                          output_dir: str,
                          img_format: str = 'pdf',
                          labels: Optional[List[str]] = None,
                          processes: Optional[int] = -1) -> List[shap.TreeExplainer]:
    """
    Generates global (model-level) explanations for the given regression models. Namely, uses `shap` library to create
    plots of feature importance, density and dependence plots for each feature.
    :param list[RegressorMixin] models: the regression models to be explained.
    :param list[pd.DataFrame] x: the feature data sets for which to extract SHAP values for each model,
    shaped (num_instances, num_features).
    :param int max_num_features: the maximum number of features for which to generate individual dependence plots.
    :param str output_dir: the directory in which to save results.
    :param str img_format: the image format of the plots to be generated.
    :param list[str] labels: the labels for each model.
    :param int processes: the number of processes used in parallel. Uses `joblib` convention.
    :rtype: list[shap.TreeExplainer]
    :return: a list of SHAP global explainers, one for each of the provided models.
    """
    if labels is None:
        labels = [f'Model {i}' for i in range(len(models))]
    logging.info(f'Plotting global explanations for {len(labels)} models...')
    args = [(models[i], x[i], max_num_features, output_dir, img_format, DIMS_PALETTE[i], labels[i], processes)
            for i in range(len(models))]
    return run_parallel(explain_model_global, args, 3, use_tqdm=True)  # TODO processes


def explain_model_global(model: xgboost.XGBRegressor,
                         x: pd.DataFrame,
                         max_num_features: int,
                         output_dir: str,
                         img_format: str = 'pdf',
                         color: str = 'red',
                         label: str = None,
                         processes: int = -1) -> shap.TreeExplainer:
    """
    Generates global (model-level) explanations for the given regression model. Namely, uses `shap` library to create
    plots of feature importance, density and dependence plots for each feature.
    :param list[RegressorMixin] model: the regression model to be explained.
    :param pd.DataFrame x: the feature data for which to extract SHAP values, shaped (num_instances, num_features).
    :param int max_num_features: the maximum number of features for which to generate individual dependence plots.
    :param str output_dir: the directory in which to save results.
    :param str img_format: the image format of the plots to be generated.
    :param str color: the color used for the feature importance plot.
    :param str label: the label for the model.
    :param int processes: the number of processes used in parallel. Uses `joblib` convention.
    :rtype: the SHAP global explainer computed for the given model and data.
    :return:
    """
    dummy_plotly()  # just to clear imports

    # create sub-dir
    _output_dir = os.path.join(output_dir, label.lower())
    create_clear_dir(_output_dir, clear=False)

    # check shap file
    shap_file = os.path.join(_output_dir, get_clean_filename(f'shap-values-{label}-df.pkl.gz'))
    shap_int_file = os.path.join(_output_dir, get_clean_filename(f'shap-interactions-{label}-np.pkl.gz'))

    # see https://slundberg.github.io/shap/notebooks/Census%20income%20classification%20with%20XGBoost.html
    explainer = shap.TreeExplainer(model)
    if os.path.isfile(shap_file):
        logging.info(f'Loading SHAP values from {shap_file}...')
        df = pd.read_pickle(shap_file)
        shap_values = df.loc[:, df.columns != EXPECTED_VALUE_COL].values  # remove expected value column
        expected_value = df[EXPECTED_VALUE_COL].values[0]
        explainer.expected_value = expected_value
        logging.info(f'Loaded SHAP values ({shap_values.shape}, mean {label}={expected_value:.2f})')
    else:
        logging.info(f'Computing SHAP values for {label}...')
        shap_values = explainer.shap_values(x)  # can take a while..
        expected_value = explainer.expected_value

        # create dataframe for shap values, shape: (datapoints, features + 1)
        df = pd.DataFrame(shap_values, columns=x.columns, index=x.index)
        df[EXPECTED_VALUE_COL] = explainer.expected_value
        df.to_pickle(shap_file)
        logging.info(f'Saved SHAP values to {shap_file} ({shap_values.shape}, mean {label}={expected_value:.2f})')

    if os.path.isfile(shap_int_file):
        logging.info(f'Loading SHAP interaction values from {shap_int_file}...')
        shap_interaction_values = joblib.load(shap_int_file)
        logging.info(f'Loaded SHAP interaction values {shap_interaction_values.shape}')
    else:
        # get batches of data and compute SHAP interaction values in parallel
        batch_idxs = np.split(np.arange(len(x)), np.arange(SHAP_BATCH_SIZE, len(x), SHAP_BATCH_SIZE))
        x_batches = [x.iloc[idxs[0]: idxs[-1] + 1] for idxs in batch_idxs]
        logging.info(f'Computing SHAP interaction values for {label}...')
        shap_int_values_batches = run_parallel(explainer.shap_interaction_values, x_batches,
                                               processes=processes, use_tqdm=False)
        shap_interaction_values = np.concatenate(shap_int_values_batches, axis=0)  # combine from batch
        joblib.dump(shap_interaction_values, shap_int_file)
        logging.info(f'Saved SHAP interaction values to {shap_int_file} ({shap_values.shape})')

    # plot shap values importance
    high_impact_feat_idxs = _plot_feature_importance(
        shap_values, expected_value, x, label, color, max_num_features, _output_dir, img_format)

    # plot shap values avg impact on dimension for most important features
    high_impact_n_feat = int(max_num_features / 2)  # only half of the top features from now on
    high_impact_feat_idxs = high_impact_feat_idxs[:high_impact_n_feat]
    high_impact_x = x.iloc[:, high_impact_feat_idxs]
    _plot_summary(shap_values[:, high_impact_feat_idxs], expected_value, high_impact_x,
                  label, len(high_impact_feat_idxs), _output_dir, img_format)

    # get shap interactions between all features
    high_impact_shap_int_vals = shap_interaction_values[:, high_impact_feat_idxs][..., high_impact_feat_idxs]
    _plot_interactions_summary(high_impact_shap_int_vals, expected_value, high_impact_x,
                               label, len(high_impact_feat_idxs), _output_dir, img_format)

    # plot shap dependence plots for most important features' interactions
    high_impact_shap_vals = shap_values[:, high_impact_feat_idxs]
    mean_interactions = np.mean(np.abs(high_impact_shap_int_vals), axis=0)
    highest_interactions = np.argsort(mean_interactions, axis=1)[:, ::-1]
    features = high_impact_x.columns
    for i, feature in enumerate(features):
        _plot_feature_shap_values(feature, expected_value, high_impact_shap_vals, high_impact_x,
                                  label, color, _output_dir, img_format)  # shap values for feature (no effects)
        j = highest_interactions[i][0]  # best interaction effect
        if j == i:
            j = highest_interactions[i][1]  # second best if best was main effect
        _plot_interaction_effects(feature, feature, expected_value, high_impact_shap_int_vals, high_impact_x,
                                  label, color, _output_dir, img_format)  # main effect
        _plot_interaction_effects(feature, features[j], expected_value, high_impact_shap_int_vals, high_impact_x,
                                  label, None, _output_dir, img_format)  # highest interaction effect

    return explainer


def _plot_feature_importance(shap_values: np.ndarray, expected_value: float, x: pd.DataFrame, label: str,
                             color: str, max_num_features: int, output_dir: str, img_format: str) -> np.ndarray:
    # average of the SHAP value magnitudes across the dataset
    mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
    df = pd.DataFrame(mean_abs_shap_values.reshape(1, -1), columns=x.columns)
    feature_idxs = np.argsort(mean_abs_shap_values)
    feature_idxs = feature_idxs[-min(max_num_features, len(feature_idxs)):][::-1]
    df = df[df.columns[feature_idxs]]
    plot_bar(df, f'Mean Absolute SHAP Values for {label.title()}',
             os.path.join(output_dir, get_clean_filename(f'mean-shap-{label}.{img_format}')),
             y_label=f'Mean Abs. SHAP Value (average impact on {label.title()}, mean={expected_value:.2f})',
             x_label='Feature', palette=[color], template=DEF_TEMPLATE,
             orientation='h', width=900, font_size=FEATURES_LABEL_FONT_SIZE)

    return feature_idxs


def _plot_summary(shap_values: np.ndarray, expected_value: float, x: pd.DataFrame, label: str,
                  max_num_features: Optional[int], output_dir: str, img_format: str, seed: int = 17):
    import matplotlib.pyplot as plt  # lazy loading
    np.random.seed(seed)  # to reproduce plots

    # density scatter plot of SHAP values for each feature
    shap_util.summary_plot(shap_values, x, show=False, max_display=max_num_features)
    plt.title(f'SHAP Values for {label.title()}', fontsize=TITLE_FONT_SIZE - 4)
    plt.xlabel(f'SHAP Value (impact on {label.title()}, mean={expected_value:.2f})', fontsize=AXES_TITLE_FONT_SIZE - 4)
    plt.yticks(fontsize=FEATURES_LABEL_FONT_SIZE)
    plt.gcf().set_figwidth(8)
    plt.savefig(os.path.join(output_dir, get_clean_filename(f'shap-density-{label}.{img_format}')), bbox_inches='tight')
    plt.close()


def _plot_interactions_summary(shap_interaction_values: np.ndarray, expected_value: float, x: pd.DataFrame,
                               label: str, max_num_features: Optional[int], output_dir: str, img_format: str,
                               seed: int = 17):
    import matplotlib.pyplot as plt  # lazy loading
    np.random.seed(seed)  # to reproduce plots

    # plot density plots of interactions
    shap_util.summary_plot(shap_interaction_values, x, max_display=max_num_features, show=False)
    plt.yticks(fontsize=FEATURES_LABEL_FONT_SIZE)
    plt.gca().tick_params('y', length=0, which='major')  # close gap between feature labels and main plot
    plt.gcf().set_figwidth(15)
    plt.savefig(os.path.join(output_dir, get_clean_filename(f'shap-interactions-{label}.{img_format}')),
                bbox_inches='tight')
    plt.close()

    # plot confusion matrix of interactions
    row_names = [f'{col}   {i}' for i, col in enumerate(x.columns)]
    col_names = [str(i) for i in range(len(x.columns))]
    df = pd.DataFrame(np.mean(np.abs(shap_interaction_values), axis=0), index=row_names, columns=col_names)
    # mult off diagonal by 2:
    # see https://slundberg.github.io/shap/notebooks/NHANES%20I%20Survival%20Model.html#Compute-SHAP-Interaction-Values
    df.where(df.values == np.diagonal(df), df.values * 2, inplace=True)
    plot_matrix(df, f'Mean Abs. SHAP Interaction Values for {label.title()}',
                os.path.join(output_dir, get_clean_filename(f'mean-shap-interactions-{label}.{img_format}')),
                palette=MATRIX_PALETTE, height=430, show_values=False, symmetrical=True,
                interpolate=False, nan_color='white')


def _plot_feature_shap_values(feature: str, expected_value: float, shap_values: np.ndarray, x: pd.DataFrame,
                              label: str, color: Optional[str], output_dir: str, img_format: str, seed: int = 17):
    import matplotlib.pyplot as plt  # lazy loading
    np.random.seed(seed)  # to reproduce plots

    # plots all shap values for the feature via dependence plots
    shap_util.dependence_plot(feature, shap_values, x, color=color, show=False, interaction_index=feature)
    # shap.dependence_plot(main_feature, shap_values, x, color=color, show=False, interaction_index=other_feature)
    plt.title(f'SHAP Values for {label.title()}', fontsize=TITLE_FONT_SIZE - 4)
    plt.xlabel(feature, fontsize=AXES_TITLE_FONT_SIZE - 4)
    plt.ylabel(f'SHAP Value\n(impact on {label.title()}, mean={expected_value:.2f})',
               fontsize=AXES_TITLE_FONT_SIZE - 4)
    plt.savefig(os.path.join(output_dir, get_clean_filename(f'shap-values-{label}-{feature}.{img_format}')),
                bbox_inches='tight')
    plt.close()


def _plot_interaction_effects(main_feature: str, other_feature: str, expected_value: float,
                              shap_int_values: np.ndarray, x: pd.DataFrame,
                              label: str, color: Optional[str], output_dir: str, img_format: str, seed: int = 17):
    import matplotlib.pyplot as plt  # lazy loading
    np.random.seed(seed)  # to reproduce plots

    # plots interaction effects via dependence plots
    shap_util.dependence_plot((main_feature, other_feature), shap_int_values, x, color=color, show=False)
    # shap.dependence_plot(main_feature, shap_values, x, color=color, show=False, interaction_index=other_feature)
    prefix = 'Main' if main_feature == other_feature else 'Interaction'
    plt.title(f'{prefix} Effects on {label.title()}', fontsize=TITLE_FONT_SIZE - 4)
    plt.xlabel(main_feature, fontsize=AXES_TITLE_FONT_SIZE - 4)
    plt.ylabel(f'SHAP Value\n(impact on {label.title()}, mean={expected_value:.2f})',
               fontsize=AXES_TITLE_FONT_SIZE - 4)
    plt.savefig(os.path.join(output_dir, get_clean_filename(f'{prefix}-effect-{label}-{main_feature}.{img_format}')),
                bbox_inches='tight')
    plt.close()


def _plot_decision(shap_values: np.ndarray, expected_value: float, x: pd.Series,
                   label: str, title: str, file_suffix: str, max_num_features: int, output_dir: str):
    import matplotlib.pyplot as plt  # lazy loading

    # save dataframe
    df = pd.DataFrame(np.array([shap_values, x.values]).T, index=x.index, columns=['SHAP Value', 'Feature Value'])
    df.sort_values('SHAP Value', key=np.abs, ascending=False)
    df.to_csv(os.path.join(output_dir, get_clean_filename(
        f'shap-force-{get_file_changed_extension(file_suffix, "csv")}')))

    plt.ion()  # non-blocking plotting
    explanation = shap.Explanation(shap_values,
                                   base_values=expected_value,
                                   data=x,
                                   feature_names=x.index,
                                   output_names=[label])
    shap_util.waterfall_plot(explanation, max_display=max_num_features, show=False)
    plt.title(f'SHAP Values for "{title}" Instance', fontsize=TITLE_FONT_SIZE - 4)
    plt.gca().xaxis.set_label_position('bottom')
    plt.xlabel(f'SHAP Value (impact on {label})', fontsize=AXES_TITLE_FONT_SIZE - 4, labelpad=40)
    plt.savefig(os.path.join(output_dir, get_clean_filename(f'shap-force-{file_suffix}')), bbox_inches='tight')
    plt.close()


def explain_diff(obs1: np.ndarray, obs2: np.ndarray, space: gym.Space, labels: List[str],
                 scaler: StandardScaler, max_num_features: int):
    # get transformed and scaled features
    obs_features = extract_features(np.array([obs1, obs2]), space, labels=labels)
    feature_labels = obs_features.columns
    obs_features = scaler.fit_transform(obs_features)
    obs1, obs2 = obs_features[0], obs_features[1]

    # get diff, sort by magnitude
    diff = obs2 - obs1
    sorted_idxs = list(reversed(np.argsort(np.abs(diff))))
    if max_num_features is not None:
        max_num_features = min(max_num_features, len(sorted_idxs))
        sorted_idxs = sorted_idxs[:max_num_features]

    # get descriptions for the features diffs from difference values and feature labels ?
