import collections
import json
import logging
import os.path
import tempfile
from typing import List, Optional, Union, Dict, OrderedDict, Tuple, Any

import numpy as np
import pandas as pd
import tqdm
import xgboost
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.mongoexp import MongoTrials
from hyperopt.pyll import scope
from plotly import express as px, graph_objs as go
from pymongo.errors import ServerSelectionTimeoutError

from ixdrl.interpretation import get_clean_filename, DIMS_PALETTE, FEATURES_LABEL_FONT_SIZE, \
    remove_nan_targets
from ixdrl.util.io import create_clear_dir, save_dict_json, get_file_changed_extension, \
    get_file_name_without_extension, save_object, load_object
from ixdrl.util.mp import run_parallel
from ixdrl.util.plot import DEF_TEMPLATE, plot_timeseries, format_and_save_plot, plot_bar, dummy_plotly

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

# https://xgboost.readthedocs.io/en/stable/python/python_api.html?highlight=XGBRegressor#xgboost.XGBRegressor
XGB_PARAM_SPACE = dict(
    max_depth=scope.int(hp.quniform('max_depth', 6, 16, 2)),
    # number of gradient boosted trees (1 added per iteration)
    n_estimators=scope.int(hp.quniform('n_estimators', 100, 500, 200)),
    colsample_bytree=hp.quniform('colsample_bytree', 0.7, 1.0, 0.1),
    min_child_weight=scope.int(hp.quniform('min_child_weight', 0, 10, 2)),
    # subsample=hp.quniform('subsample', 0.7, 1.0, 0.1),
    learning_rate=hp.quniform('learning_rate', 0.1, 0.3, 0.1),
    # gamma=hp.choice('gamma', np.arange(0, 20, 0.5, dtype=float)),
    reg_alpha=hp.quniform('reg_alpha', 0., 5., 0.5),
    # reg_lambda=hp.quniform('reg_lambda', 0., 20., 1.)
)

XGB_EVAL_METRICS = ['mae', 'rmse']
XGB_MAIN_METRIC = 'mae'
XGB_IMPORTANCE_TYPE = 'gain'
XGB_EARLY_STOPPING = 20  # 30 # number of iterations without improvement for early stopping
XGB_VERBOSITY = 1

HPO_MAX_ITERATIONS = 200  # 500  # 200  # number of hyper-param optimization iterations
HPO_EARLY_STOPPING = 20  # 50  # number of hyper-param optimization iterations without improvement
HPO_OPTIM_ALGO = tpe.suggest
HPO_MONGO_PORT = 1234  # port where Mongo DB server is listening
HPO_MONGO_DB_NAME = 'ixdrl'


def create_regression_model(processes: int = -1,
                            seed: int = 17,
                            eval_metrics: List[str] = XGB_EVAL_METRICS,
                            early_stopping_rounds: int = XGB_EARLY_STOPPING,
                            missing_value: float = np.nan,
                            **kwargs) -> xgboost.XGBRegressor:
    """
    Creates an XGBoost regression model.
    :param int processes: the number of processes to use to train the models (if needed). Follows `joblib` convention.
    :param int seed: the seed to initialize the random number generation (for reproducibility).
    :param list[str] eval_metrics: metric(s) used for monitoring the training result and early stopping.
    :param int early_stopping_rounds: number of iterations without improvement needed to activate early stopping.
    :param float missing_value: value in the data which needs to be present as a missing value.
    :param kwargs: the extra parameters for the XGBoost model, possibly optimized.
    :rtype: xgboost.XGBRegressor
    :return: the parameterized XGBoost model.
    """
    # comparison of sklearn options for regression model: https://airtable.com/shrQ5rfksG64QobIy/tblkAQbukd3Al0OT6
    # XGBoost supports missing values: https://machinelearningmastery.com/handle-missing-data-python/
    return xgboost.XGBRegressor(n_jobs=processes,
                                random_state=seed,
                                missing=missing_value,
                                # importance_type=IMPORTANCE_TYPE,
                                eval_metric=eval_metrics,
                                early_stopping_rounds=early_stopping_rounds,
                                verbosity=XGB_VERBOSITY,
                                **kwargs)


def select_train_test_split(x: List[pd.DataFrame],
                            y: List[pd.DataFrame],
                            test_fraction: float,
                            labels: Optional[List[str]] = None,
                            seed: int = 17) -> \
        Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Randomly selects train and test data to be used for training and evaluation of each regression model.
    :param pd.DataFrame x: the datasets with the features (columns) for each data instance (rows) per model.
    :param pd.DataFrame y: the datasets with a target (column) for each data instance (rows) per model.
    The `x` and `y` datasets are assumed to be index-aligned.
    :param float test_fraction: the fraction of the data to be used for testing.
    :param list[str] labels: the label/name of each model to be trained.
    :param int seed: the seed to initialize the random split number generation (for reproducibility).
    :rtype: tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]
    :return: a tuple containing the feature train, target train, feature test and target test sets for each model.
    """
    assert len(x) == len(y), f'Size of features set ({len(x)}) does not match size of targets set ({len(y)})'

    # select train/test indices for each model according to available data
    logging.info('_________________________________________')
    logging.info(f'Selecting train and test data for {len(x)} models...')
    x_train, y_train, x_test, y_test = [], [], [], []
    for i in tqdm.tqdm(range(len(x))):
        num_instances = len(x[i])
        all_idxs = np.arange(num_instances)
        test_idxs = sorted(np.random.RandomState(seed).choice(
            all_idxs, int(test_fraction * num_instances), replace=False))
        train_idxs = sorted(set(all_idxs) - set(test_idxs))
        x_train.append(x[i].iloc[train_idxs])
        y_train.append(y[i].iloc[train_idxs])
        x_test.append(x[i].iloc[test_idxs])
        y_test.append(y[i].iloc[test_idxs])
        logging.info(f'{labels[i]}: {len(train_idxs)} training, {len(test_idxs)} test instances')

    return x_train, y_train, x_test, y_test


def load_models(paths: List[str], labels: Optional[List[str]] = None) -> List[Optional[xgboost.XGBRegressor]]:
    """
    Loads the regression models from the given paths. If the file for a model does not exist, it's index is returned.
    :param list[str] paths: the list of paths to load each model from.
    :param list[str] labels: the label/name of each model to be loaded.
    :rtype: list[xgboost.XGBRegressor]
    :return: a list the same size as `paths` containing either the loaded models from the given paths or `None`, in
    which case a model could not be found in the corresponding path.
    """
    if labels is None:
        labels = [get_file_name_without_extension(path) for path in paths]
    assert len(paths) == len(labels), \
        f'Number of model paths ({len(paths)}) does not match number of model labels ({len(labels)})'

    logging.info(f'Trying to load {len(paths)} regression models from file...')
    models: List[Optional[xgboost.XGBRegressor]] = []
    for i, model_path in tqdm.tqdm(enumerate(paths), total=len(paths)):
        if os.path.isfile(model_path):
            # creates generic model and loads tree from file
            logging.info(f'Loading estimator for {labels[i]} from {model_path}...')
            model = create_regression_model()
            model.load_model(model_path)
            models.append(model)
        else:
            models.append(None)
    return models


def train_models(needs_train_idxs: List[int],
                 x_train: List[pd.DataFrame],
                 y_train: List[pd.DataFrame],
                 x_test: List[pd.DataFrame],
                 y_test: List[pd.DataFrame],
                 model_paths: List[str],
                 labels: Optional[List[str]] = None,
                 processes: int = -1,
                 seed: int = 17,
                 eval_metrics: List[str] = XGB_EVAL_METRICS,
                 early_stopping_rounds: int = XGB_EARLY_STOPPING,
                 missing_value: float = np.nan) -> List[xgboost.XGBRegressor]:
    """
    Trains regression models using the provided features->targets pairs, saves the models to file.
    :param list[int] needs_train_idxs: the indices of the models to be trained. Models whose index is not on the list
    will remain the same.
    :param pd.DataFrame x_train: the train datasets with the features (columns) for each data instance (rows) per model.
    :param pd.DataFrame y_train: the train datasets with a target (column) for each data instance (rows) per model.
    The `x_train` and `y_train` datasets are assumed to be index-aligned.
    :param pd.DataFrame x_test: the test datasets with the features (columns) for each data instance (rows) per model.
    :param pd.DataFrame y_test: the test datasets with a target (column) for each data instance (rows) per model.
    The `x_test` and `y_test` datasets are assumed to be index-aligned.
    :param list[str] model_paths: the file paths in which to save each regression model.
    :param list[str] labels: the label/name of each model to be trained.
    :param int processes: the number of processes to use to train the models (if needed). Follows `joblib` convention.
    :param int seed: the seed to initialize the random number generation (for reproducibility).
    :param list[str] eval_metrics: metric(s) used for monitoring the training result and early stopping.
    :param int early_stopping_rounds: number of iterations without improvement needed to activate early stopping.
    :param float missing_value: value in the data which needs to be present as a missing value.
    :rtype: list[xgboost.XGBRegressor]
    :return: the list of trained models.
    """
    assert len(x_train) == len(y_train) == len(x_test) == len(y_test), \
        'Inconsistent number of feature datasets compared to target datasets'

    logging.info('_________________________________________')
    labels = [labels[i] for i in needs_train_idxs]
    labels_str = '\n\t'.join(labels)
    logging.info(f'Training {len(needs_train_idxs)} regression models:\n\t{labels_str}\n\t...')

    # train and save models
    args = [(x_train[i], y_train[i], x_test[i], y_test[i], model_paths[i],
             processes, seed, eval_metrics, early_stopping_rounds, missing_value)
            for i in needs_train_idxs]
    return run_parallel(fit_estimator, args, processes, use_tqdm=False)


def evaluate_models(models: List[xgboost.XGBRegressor],
                    model_paths: List[str],
                    x: List[pd.DataFrame],
                    y: List[pd.DataFrame],
                    output_dir: str,
                    max_num_features: int,
                    labels: Optional[List[str]] = None,
                    img_format: str = 'pdf'):
    """
    Evaluates the given trained models by plotting training results and performing regression evaluation on the given
    test set. It also performs internal feature importance analysis using XGBoost.
    :param list[xgboost.XGBRegressor] models: the trained models to be evaluated.
    :param list[str] model_paths: the paths to each model file (where the model params and hyperopt trials might as
    well be stored).
    :param pd.DataFrame x: the test datasets containing the features (columns) for each data instance (rows) per model.
    :param pd.DataFrame y: the test datasets containing a target (column) for each data instance (rows) per model.
    The `x` and `y` datasets are assumed to be index-aligned.
    :param str output_dir: the directory in which to save regression evaluation results.
    :param int max_num_features: the maximum number of features for feature importance analysis.
    :param list[str] labels: the label/name of each model to be trained.
    :param str img_format: the image format of the plots to be generated.
    """
    dummy_plotly()  # just to clear imports

    logging.info('_________________________________________')
    logging.info('Plotting hyper-param optimization results...')
    _output_dir = os.path.join(output_dir, 'hyperparam-optimization')
    create_clear_dir(_output_dir, clear=False)
    _plot_hyperparam_optim(model_paths, labels, _output_dir, img_format)
    _plot_best_hyperparams(model_paths, labels, _output_dir, img_format)
    logging.info(f'Saved hyper-param optimization results to {_output_dir}')

    logging.info('_________________________________________')
    logging.info('Evaluating the regression model...')
    _output_dir = os.path.join(output_dir, 'regression-eval')
    create_clear_dir(_output_dir, clear=False)
    _plot_training_evaluation(models, labels, _output_dir, img_format)
    _plot_regression_results(models, x, y, _output_dir, img_format)
    logging.info(f'Saved regression evaluation results to {_output_dir}')

    logging.info('_________________________________________')
    logging.info('Performing feature importance analysis...')
    _output_dir = os.path.join(output_dir, 'model-feature-importance')
    create_clear_dir(_output_dir, clear=False)
    _plot_feature_importance(models, labels, max_num_features, _output_dir, img_format)
    logging.info(f'Saved feature importance results to {_output_dir}')


def fit_estimator(x_train: pd.DataFrame,
                  y_train: pd.DataFrame,
                  x_test: pd.DataFrame,
                  y_test: pd.DataFrame,
                  model_path: str,
                  processes: int = -1,
                  seed: int = 17,
                  eval_metrics: List[str] = XGB_EVAL_METRICS,
                  early_stopping_rounds: int = XGB_EARLY_STOPPING,
                  missing_value: float = np.nan) \
        -> xgboost.XGBRegressor:
    """
    Trains a regression model using the provided data using hyper-parameter optimization.
    :param pd.DataFrame x_train: the train dataset containing the features (columns) for each data instance (rows).
    :param pd.DataFrame y_train: the train dataset containing the target (column) for each data instance (rows).
    The `x_train` and `y_train` datasets are assumed to be index-aligned.
    :param pd.DataFrame x_test: the test dataset containing the features (columns) for each data instance (rows).
    :param pd.DataFrame y_test: the test dataset containing the target (column) for each data instance (rows).
    The `x_test` and `y_test` datasets are assumed to be index-aligned.
    :param str model_path: the path to the file in which to save the model once trained.
    :param int processes: the number of processes to use to train the models (if needed). Follows `joblib` convention.
    :param int seed: the seed to initialize the random number generation (for reproducibility).
    :param list[str] eval_metrics: metric(s) used for monitoring the training result and early stopping.
    :param int early_stopping_rounds: number of iterations without improvement needed to activate early stopping.
    :param float missing_value: value in the data which needs to be present as a missing value.
    :rtype: xgboost.XGBRegressor
    :return: the XGBoost regression model fitted to the given train data.
    """
    # select a specific target from the columns
    x_train, y_train = remove_nan_targets(x_train, y_train)
    x_test, y_test = remove_nan_targets(x_test, y_test)

    temp_dir = tempfile.TemporaryDirectory()  # temp directory to store intermediate models

    def _no_progress_loss(_trials, best_loss=None, iteration_no_progress=0):
        # checks hyperopt progress
        if _trials.trials[-1]['result']['status'] == 'new':
            return False, [None, 0]  # see: https://github.com/hyperopt/hyperopt/issues/808#issuecomment-926283421
        new_loss = _trials.trials[len(_trials.trials) - 1]['result']['loss']
        if best_loss is None:
            return False, [new_loss, iteration_no_progress + 1]
        if new_loss < best_loss:
            best_loss = new_loss
            iteration_no_progress = 0  # improvement, reset
        else:
            iteration_no_progress += 1  # no improvement, increase
        return iteration_no_progress >= HPO_EARLY_STOPPING, [best_loss, iteration_no_progress]

    def score(model_params):
        # create model from params and fit
        _model = create_regression_model(processes, seed, eval_metrics, early_stopping_rounds, missing_value,
                                         **model_params)
        _model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=True)

        # final score is min of the best metric for test set
        _score = np.min(_model.evals_result()['validation_1'][XGB_MAIN_METRIC])

        # save model to temp_dir
        temp_file = tempfile.NamedTemporaryFile(dir=temp_dir.name, suffix='.ubj', delete=False).name
        _model.save_model(temp_file)

        return dict(loss=_score, status=STATUS_OK, model_path=temp_file)

    # runs hyper-param optimization parallelized via Mongo DB
    logging.info('________________________________________')
    conn_str = f'mongo://localhost:{HPO_MONGO_PORT}/{HPO_MONGO_DB_NAME}/jobs'
    try:
        # try parallel search via Mongo DB server
        logging.info(f'Trying to connect to Mongo DB server for parallel hyperopt at: {conn_str}')
        trials = MongoTrials(conn_str, exp_key=f'feature_importance_{y_test.columns[0]}')
        trials.handle.collection.delete_many({})
        trials.delete_all()
        logging.info('Connection successful to Mongo DB server')
    except ServerSelectionTimeoutError:
        trials = Trials()  # fall back to non-parallel search
        logging.info('Failed; using non-parallel hyperopt')

    # logging.info('________________________________________')
    best_params = fmin(score, XGB_PARAM_SPACE,
                       trials=trials,
                       algo=HPO_OPTIM_ALGO,
                       max_evals=HPO_MAX_ITERATIONS,
                       early_stop_fn=_no_progress_loss)

    # tries to load best model from hyperopt results
    # TODO still fits a new model using loaded params since otherwise the models gets very large?
    best_path = trials.results[trials.best_trial['tid']]['model_path']
    print(best_path)
    if os.path.isfile(best_path):
        model = create_regression_model()
        model.load_model(best_path)
        best_params = model.get_xgb_params()
        del best_params['eval_metric']
        del best_params['n_jobs']
        del best_params['random_state']
        del best_params['verbosity']
        best_params['n_estimators'] = model.n_estimators  # not in the params..
    else:
        print('NO MODEL FILE!!')

    print(best_params)

    # creates model from best params and fits it
    model = create_regression_model(processes=processes,
                                    seed=seed,
                                    eval_metrics=eval_metrics,
                                    early_stopping_rounds=early_stopping_rounds,
                                    missing_value=missing_value,
                                    **space_eval(XGB_PARAM_SPACE, best_params))
    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)], verbose=True)

    model.save_model(model_path)  # saves model to file

    # gets and saves all model params
    params = model.get_xgb_params()
    params.update(space_eval(XGB_PARAM_SPACE, best_params))
    params = collections.OrderedDict({k: params[k] for k in sorted(params.keys())})
    save_dict_json(params, get_file_changed_extension(model_path, 'json', suffix='-params'))

    # saves trials object with the optimization results
    save_object(trials, get_file_changed_extension(model_path, 'pkl', suffix='-trials'), compress_gzip=False)

    # delete temp dir
    temp_dir.cleanup()

    return model  # return best model


def _get_params_from_trial(trial: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gets the param values from the indices stored in given trial object.
    """
    return space_eval(XGB_PARAM_SPACE, {k: v[0] for k, v in trial['misc']['vals'].items()})


def _plot_hyperparam_optim(model_paths: List[str], dims: List[str],
                           output_dir: str, img_format: str, template: str = DEF_TEMPLATE):
    logging.info(f'Plotting regression model hyperparam optimization data for {len(dims)} dimensions...')
    for i, model_path in tqdm.tqdm(enumerate(model_paths), total=len(model_paths)):
        params_losses = {p: {} for p in XGB_PARAM_SPACE.keys()}
        trials_file = get_file_changed_extension(model_path, 'pkl', suffix='-trials')
        if os.path.isfile(trials_file):
            for trial in load_object(trials_file):
                if trial['result']['status'] == STATUS_OK:
                    params = _get_params_from_trial(trial)
                    loss = trial['result']['loss']
                    for p, v in params.items():
                        if v not in params_losses[p]:
                            params_losses[p][v] = []
                        params_losses[p][v].append(loss)  # add loss to param value list

        _output_dir = os.path.join(output_dir, dims[i].lower())
        create_clear_dir(_output_dir, clear=False)

        for p, losses in params_losses.items():
            if len(losses) == 0:
                continue
            # plots mean loss associated with each param value for the current dim model
            df = pd.DataFrame(dict([(k, pd.Series(losses[k])) for k in sorted(losses.keys())])).T
            plot_timeseries(df, f'Mean Loss for {dims[i].title()}',
                            os.path.join(_output_dir, f'{p.lower()}.{img_format}'), palette=[DIMS_PALETTE[i]],
                            average=True, x_label=p, y_label='Loss', show_legend=False,
                            xaxis_range=[df.index.min(), df.index.max()], markers=True, template=template)


def _plot_best_hyperparams(model_paths: List[str], dims: List[str],
                           output_dir: str, img_format: str, template: str = DEF_TEMPLATE):
    logging.info(f'Collecting regression model parameters for {len(dims)} dimensions...')
    all_params = {p: [] for p in XGB_PARAM_SPACE.keys()}
    for model_path in tqdm.tqdm(model_paths):
        params_file = get_file_changed_extension(model_path, 'json', suffix='-params')
        if os.path.isfile(params_file):
            with open(params_file, 'r') as fp:
                params = json.load(fp)
            for p, v in params.items():
                if p in all_params:
                    all_params[p].append(v)

    logging.info(f'Plotting values for all {len(all_params)} parameters...')
    dims = [d.title() for d in dims]
    for p, vals in tqdm.tqdm(all_params.items(), total=len(all_params)):
        if len(vals) == len(dims):
            plot_bar(pd.DataFrame(dict(zip(dims, vals)), index=[0]), f'Parameter: {p}',
                     os.path.join(output_dir, f'best-{p}.{img_format}'),
                     x_label='Dimension', y_label='Param. value', palette=DIMS_PALETTE,
                     template=template, plot_mean=True)


def _plot_training_evaluation(models: List[xgboost.XGBRegressor], dims: List[str],
                              output_dir: str, img_format: str, template: str = DEF_TEMPLATE):
    validation_names = {'validation_0': 'train', 'validation_1': 'test'}
    logging.info(f'Plotting evaluation during training for {len(dims)} dimensions...')
    metrics_scores: Dict[str, OrderedDict[str, float]] = {}
    for i, model in tqdm.tqdm(enumerate(models), total=len(models)):
        for val_name, metric_scores in model.evals_result().items():
            val_name = validation_names[val_name].lower()
            for metric, scores in metric_scores.items():
                plot_timeseries(
                    pd.DataFrame(scores),
                    f'{dims[i].title()} Regression Training Results ({metric.upper()} on {val_name} set)',
                    os.path.join(output_dir,
                                 get_clean_filename(f'evolution-{val_name}-{metric}-{dims[i]}.{img_format}')),
                    x_label='Iteration', y_label=metric.upper(), show_legend=False,
                    palette=[DIMS_PALETTE[i]], template=template)

                if val_name == 'test':
                    if metric not in metrics_scores:
                        metrics_scores[metric] = collections.OrderedDict({})
                    metrics_scores[metric][dims[i]] = np.min(scores)  # select best score on test set

    logging.info(f'Plotting evaluation results on the test set for {len(metrics_scores)} metrics...')
    for metric, metric_scores in metrics_scores.items():
        plot_bar(pd.DataFrame(metric_scores, index=[0]), f'Regression Evaluation on Test Set ({metric.upper()})',
                 os.path.join(output_dir, get_clean_filename(f'test-score-{metric}.{img_format}')),
                 x_label='Dimension', y_label=metric.upper(), palette=DIMS_PALETTE, template=template,
                 plot_mean=True)


def _plot_regression_results(models: List[xgboost.XGBRegressor], x_test: List[pd.DataFrame], y_test: List[pd.DataFrame],
                             output_dir: Optional[str], img_format: str, template: str = DEF_TEMPLATE):
    dims = [y_test[i].columns[0] for i in range(len(y_test))]
    logging.info(f'Saving regression results for {len(dims)} dimensions...')
    for i, model in tqdm.tqdm(enumerate(models), total=len(models)):
        # scatter plot with predicted vs GT targets
        y_pred = models[i].predict(x_test[i])
        output_img = os.path.join(output_dir, get_clean_filename(f'regression-eval-{dims[i]}.{img_format}'))
        title = f'{dims[i].title()} Regression Model Evaluation'
        df = pd.DataFrame(np.array([y_test[i].values.flatten(), y_pred]).T, columns=['Ground Truth', 'Predicted'])
        fig = px.scatter(df, x='Ground Truth', y='Predicted',
                         color_discrete_sequence=[DIMS_PALETTE[i]], title=title, template=template)
        fig.add_trace(go.Scatter(x=[np.min(df.values), np.max(df.values)], y=[np.min(df.values), np.max(df.values)],
                                 mode='lines', line=dict(width=2, dash='dash', color='red')))
        format_and_save_plot(fig, df, title, output_img, width=600, height=600, show_legend=False)


def _plot_feature_importance(models: List[xgboost.XGBRegressor], dims: List[str], max_num_features: int,
                             output_dir: str, img_format: str, template: str = DEF_TEMPLATE):
    logging.info(f'Plotting importance for {len(dims)} dimensions...')
    for i, model in tqdm.tqdm(enumerate(models), total=len(models)):
        for metric in ['weight', 'cover', 'gain']:
            _plot_importance(model, f'Importance of Features\' {metric.title()} for {dims[i].title()}',
                             os.path.join(output_dir, get_clean_filename(f'{dims[i]}-{metric}.{img_format}')),
                             DIMS_PALETTE[i], metric, max_num_features=max_num_features, template=template)


def _plot_importance(model: xgboost.XGBRegressor, title: str, output_img: str, color: Union[str],
                     importance_type: str = XGB_IMPORTANCE_TYPE, max_num_features: Optional[int] = None,
                     template: str = DEF_TEMPLATE):
    # based on xgboost.plotting.plot_importance to work with plotly
    importance = model.get_booster().get_score(importance_type=importance_type)
    df = pd.DataFrame(importance, index=[0])
    df = df[sorted(df.columns, key=lambda col: df.loc[0, col])]
    if max_num_features is not None:
        df = df.iloc[:, :max_num_features]
    plot_bar(df, title, output_img, x_label='Features', y_label='F score', palette=[color], template=template,
             orientation='h', width=900, font_size=FEATURES_LABEL_FONT_SIZE)
