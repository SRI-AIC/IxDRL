import colorsys
import os
import tempfile
import warnings
from enum import IntEnum
from typing import Union, List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import scipy.stats as st

from .io import get_file_changed_extension
from .math import stretch_array, weighted_nanmean, weighted_nanstd

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

DEF_PALETTE = px.colors.colorbrewer.Set1
DEF_HIST_PALETTE = px.colors.colorbrewer.Spectral
DEF_MATRIX_PALETTE = 'Sunset'
DEF_TEMPLATE = 'plotly_white'  # 'plotly_dark', 'presentation',
ERROR_AREA_ALPHA = 0.3
ERROR_AREA_LINE_WIDTH = 0
ERROR_BAR_THICKNESS = 1
ERROR_BAR_WIDTH = 6
MARKER_SIZE = 10
RADAR_ALPHA = 0.4
ALL_FONT_SIZE = 16
AXES_TITLE_FONT_SIZE = 18
TITLE_FONT_SIZE = 24


class ErrorStat(IntEnum):
    """
    Types of error to be plotted as shaded areas in timeseries plots, or error bars in bar plots.
    """
    StdDev = 0
    StdError = 1
    CI95 = 2
    CI99 = 3


def _get_mean_error_stat(data: Union[np.ndarray, pd.DataFrame],
                         error_stat: ErrorStat,
                         group_by: Optional[str] = None,
                         weights: Optional[Union[np.ndarray, str]] = None,
                         keepdims: bool = True,
                         axis: int = None) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[pd.Series, pd.Series]]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)  # avoid empty slice warnings

        # gets mean std dev according to data type
        if isinstance(data, np.ndarray) and (weights is None or isinstance(weights, np.ndarray)):
            mean = weighted_nanmean(data, weights=weights, axis=axis, keepdims=keepdims)
            std = weighted_nanstd(data, weights=weights, axis=axis, keepdims=keepdims)
            n = data.shape[axis or 0]
        elif isinstance(data, pd.DataFrame) and (weights is None or isinstance(weights, str)):
            if group_by is not None:
                mean_dfs: List[pd.DataFrame] = []
                std_dfs: List[pd.DataFrame] = []
                for g, _df in data.groupby(group_by, sort=False):
                    # compute mean and std for each group
                    if group_by in _df.columns:
                        _df = _df.drop(group_by, axis=1)
                    mean_df, std_df = _get_mean_error_stat(
                        _df, error_stat=error_stat, weights=weights, keepdims=keepdims, axis=axis)
                    mean_df[group_by] = g
                    mean_df.set_index(group_by, inplace=True)  # group label is set as index
                    std_df[group_by] = g
                    std_df.set_index(group_by, inplace=True)  # group label is set as index
                    mean_dfs.append(mean_df)
                    std_dfs.append(std_df)
                mean_df = pd.concat(mean_dfs, axis=0)
                std_df = pd.concat(std_dfs, axis=0)
                return mean_df, std_df

            if weights is not None:
                # get weights and drop weights column
                _weights = data[weights].values.reshape(-1, 1)
                data.drop(weights, axis=1, inplace=True)
                weights = _weights

            mean = pd.DataFrame(weighted_nanmean(data.values, weights=weights, keepdims=keepdims, axis=axis),
                                columns=data.columns if axis == 0 else None)
            std = pd.DataFrame(weighted_nanstd(data.values, weights=weights, keepdims=keepdims, axis=axis),
                               columns=data.columns if axis == 0 else None)
            n = data.shape[axis or 0]
        else:
            raise ValueError(f'Data type not supported: {data}, or weights incompatible with data: {weights}')

        # gets error stat
        if error_stat == ErrorStat.StdDev:
            return mean, std
        if error_stat == ErrorStat.StdError:
            return mean, std / np.sqrt(n)
        if error_stat == ErrorStat.CI95 or error_stat == ErrorStat.CI99:
            # gets the confidence interval (CI)
            # see: https://vedexcel.com/how-to-calculate-confidence-intervals-in-python/
            alpha = 0.95 if error_stat == ErrorStat.CI95 else 0.99
            if n < 30:
                ci = st.t.interval(confidence=alpha, df=n - 1, loc=mean, scale=std / np.sqrt(n))
            else:
                ci = st.norm.interval(confidence=alpha, loc=mean, scale=std / np.sqrt(n))
            return mean, np.abs(mean - ci[0])
        raise ValueError(f'Unknown error statistic: {error_stat}')


def _convert_data(data: Union[pd.DataFrame, pd.Series, Dict[str, np.ndarray]]) -> pd.DataFrame:
    if isinstance(data, dict):
        data = pd.DataFrame.from_dict(data, orient='index').transpose()
    elif isinstance(data, pd.Series):
        data = data.to_frame()
    elif not isinstance(data, pd.DataFrame):
        raise ValueError(f'Unknown data type provided: {data}')
    return data


def plot_timeseries(data: Union[pd.DataFrame, Dict[str, np.ndarray]],
                    title: Optional[str] = None,
                    output_img: Optional[str] = None,
                    save_csv: bool = True,
                    save_json: bool = True,
                    x_label: Optional[str] = None,
                    y_label: Optional[str] = None,
                    var_label: Optional[str] = None,
                    palette: Optional[Union[List[str], str]] = DEF_PALETTE,
                    template: str = DEF_TEMPLATE,
                    width: Optional[int] = 600,
                    height: Optional[int] = 450,
                    plot_mean: bool = False,
                    normalize_samples: Optional[int] = None,
                    average: bool = False,
                    plot_avg_error: bool = True,
                    error_stat: ErrorStat = ErrorStat.CI95,
                    group_by: str = None,
                    weights: Optional[Union[pd.DataFrame, Dict[str, np.ndarray]]] = None,
                    reverse_x_axis: bool = False,
                    y_min: Optional[float] = None,
                    y_max: Optional[float] = None,
                    log_y: bool = False,
                    smooth_factor: float = 0.,
                    markers: bool = False,
                    show_legend: bool = True,
                    show_plot: bool = False,
                    **kwargs) -> go.Figure:
    """
    Plots several time-series in the same line plot, representing the "evolution" of some variables over time.
    :param pd.DataFrame data: the object containing the data to be plotted. If a `dict` is given, then each key
    corresponds to the name of a variable, while the values are arrays of shape (steps, ) containing the variables'
    value over time. If a `DataFrame` is provided, then it is assumed each column represents a different variable, while
    the indices represent different timesteps, i.e., corresponding to long-form data.
    :param str title: the title of the plot.
    :param str output_img: the path to the image file in which to save the plot.
    :param bool save_csv: whether to save the (possibly transformed) data in a CSV file.
    :param bool save_json: whether to save the plot data to a Json file for later retrieval/loading.
    :param str x_label: the label of the X axis.
    :param str y_label: the label of the Y axis.
    :param str var_label: the name given to the group of variables to be plotted.
    :param str or list[str] palette: the name of the Plotly palette used to color each series line, or a list containing
    a string representation for the colors to be used. See: https://plotly.com/python/builtin-colorscales/.
    :param str template: the name of the Plotly layout template to be used. Defaults to "plotly_white".
    See: https://plotly.com/python/templates/.
    :param int width: the plot's width.
    :param int height: the plot's height.
    :param bool plot_mean: whether to plot the "mean" series line, corresponding to the average of the series values at
    each discrete step. If only one series is plotted, then an horizontal line with the mean value is added to the plot.
    :param bool normalize_samples: the number of samples used to normalize the lengths of the different series. This
    means that the lengths of all series will be normalized between 0 and 1, and then `normalize_samples` will be
    sampled from each series to create the plot, where a series index/step value closest to the fraction over all steps
    is chosen to be plotted at each normalized interval step. The length is given by the time index starting from which
    a series only has `np.nan` values.
    A negative value of `None` means no length normalization is performed.
    :param bool average: whether to average the series before plotting.
    :param bool plot_avg_error: if `average` is `True`, this controls whether the standard error is plotted in addition
    to the mean series. This corresponds to a shaded area above/below the mean values.
    :param bool error_stat: the statistic used to plot the error area.
    :param str group_by: controls whether the data should be grouped by some column prior to plotting. This will create
    `N*M` new variables/series, one for each possible value of the `group_by` column, for each of the other variables/
    columns in the data. If `average` is `True`, then the average is performed by original variable across the grouped
    data and not over all variables.
    :param str weights: a structure equivalent to `data` containing the weights for each step of each timeseries. If a
    single row (a single value per timeseries) is provided, then all timesteps of each series are weighted equally.
    :param bool reverse_x_axis: whether to reverse the order of values in the x-axis.
    :param float y_min: the minimal value of the y-axis.
    :param float y_max: the maximal value of the y-axis.
    :param bool log_y: whether the y-axis is log-scaled in cartesian coordinates.
    :param float smooth_factor: the smoothing factor in [0, 1[ for the timeseries curves using exponential weighted mean.
    :param bool markers: whether to show markers in addition to lines at points in which there is data.
    :param bool show_legend: whether to show the plot's legend.
    :param bool show_plot: whether to show the plot, in which case a new browser tab would be opened displaying the
    interactive Plotly plot.
    :param kwargs: extra arguments passed to `format_and_save_plot`.
    :rtype: go.Figure
    :return: the Plotly figure created with the given data.
    """
    data = _convert_data(data)  # transform to pandas dataframe first

    if group_by is not None:
        assert group_by in data.columns, f'Group by column {group_by} is not a valid column: {data.columns}'
    if weights is not None:
        if isinstance(weights, dict):
            weights = pd.DataFrame.from_dict(weights, orient='index').transpose()
        assert len(set(data.columns) - set(weights.columns)) == 0, \
            f'Weights columns {weights.columns} are missing some data columns: {data.columns}'
        assert weights.shape == data.shape or weights.shape[0] == 1, \
            f'Weights should be the same size of data or have a single row, but {weights.shape} was provided'

    # group by given column
    var_groups = None
    if group_by is not None:
        var_groups = {v: [] for v in data.columns if v != group_by}  # create index for each non-group_by column
        dfs: List[pd.DataFrame] = []  # the dataframes for each group_by partition
        weight_dfs: List[pd.DataFrame] = []
        for g, g_df in data.groupby(group_by):
            g_df = g_df.drop(group_by, axis=1)  # drop the group-by column
            g_df = g_df.rename(columns={v: f'{group_by}-{g}-{v}'
                                        for v in g_df.columns})  # renames columns for this group partition
            # g_df.reset_index(drop=True, inplace=True)
            dfs.append(g_df)

            for v in var_groups.keys():
                var_groups[v].append(f'{group_by}-{g}-{v}')  # store the new columns for this group partition

            if weights is not None:
                w_df = weights.iloc[g_df.index]
                w_df = w_df.rename(columns={v: f'{group_by}-{g}-{v}'
                                            for v in w_df.columns})  # renames columns for this group partition
                # w_df.reset_index(drop=True, inplace=True)
                weight_dfs.append(w_df)

        data = pd.concat(dfs, axis=1)
        if weights is not None:
            weights = pd.concat(weight_dfs, axis=1)

    # get average and error data
    err_data: Optional[pd.DataFrame] = None
    if average:
        if var_groups is not None:
            # average each variable group
            _avg_data: Dict[str, np.ndarray] = {}
            _err_data: Dict[str, np.ndarray] = {}
            for v, cols in var_groups.items():
                _weights = None if weights is None else weights[cols].values
                _avg_data[v], _err_data[v] = _get_mean_error_stat(
                    data[cols].values, error_stat, weights=_weights, axis=1, keepdims=False)
            err_data = pd.DataFrame(_err_data, index=data.index)
            data = pd.DataFrame(_avg_data, index=data.index)
        else:
            # single timeseries (average of all timeseries)
            label = var_label or 'variable'
            _weights = None if weights is None else weights.values
            mean, err_data = _get_mean_error_stat(data.values, error_stat, weights=_weights, axis=1)
            err_data = pd.DataFrame({label: err_data.flatten()}, index=data.index)
            data = pd.DataFrame({label: mean.flatten()}, index=data.index)

    # smooth the curves
    if smooth_factor > 0:
        data = data.ewm(alpha=1 - smooth_factor).mean()
        if err_data is not None:
            err_data = err_data.ewm(alpha=1 - smooth_factor).mean()

    # checks length normalization
    if normalize_samples is not None and normalize_samples >= 1:
        x_label = 'Normalized Index' if x_label is None else x_label + ' (normalized)'
        norm_data = {}
        norm_err_data = {}
        for v in data.columns:
            non_nan = np.cumsum(~np.isnan(data[v].values))
            max_idx, = np.where(non_nan == non_nan[-1])
            max_idx = max_idx[0] + 1
            norm_data[v] = stretch_array(data[v].values[:max_idx], normalize_samples)
            if err_data is not None:
                norm_err_data[v] = stretch_array(err_data[v].values[:max_idx], normalize_samples)
        data = pd.DataFrame(norm_data, index=np.linspace(0, 1, normalize_samples))
        err_data = pd.DataFrame(norm_err_data, index=np.linspace(0, 1, normalize_samples))

    # checks color palette, discretize if needed
    colors = palette
    if isinstance(colors, str) and hasattr(px.colors.qualitative, colors):
        colors = getattr(px.colors.qualitative, colors)  # get discrete color sequence by name
    num_colors = len(data.columns)
    if isinstance(colors, str) or (isinstance(colors, list) and num_colors > len(colors) > 1):
        colors = px.colors.sample_colorscale(colors, np.linspace(0, 1, num_colors))

    # plots data, see https://plotly.com/python/wide-form/#wideform-defaults
    fig = px.line(data, title=title,
                  labels={'index': x_label or 'Index',
                          'value': y_label or 'Value',
                          'variable': var_label or 'Variable'},
                  log_y=log_y, template=template, color_discrete_sequence=colors, markers=markers)

    # plots average (std) error via shaded area
    if average and plot_avg_error:
        for shape in fig.data:
            non_nan = np.cumsum(~np.isnan(shape.y))
            max_idx, = np.where(non_nan == non_nan[-1])
            max_idx = max_idx[0] + 1
            x = shape.x[:max_idx]
            y = shape.y[:max_idx]
            err = err_data[shape.name].values[:max_idx]
            x = x[~np.isnan(y)]
            err = err[~np.isnan(y)]  # remove where y=nan to avoid incorrect error region
            err[np.isnan(err)] = 0  # also set points with undefined error to 0
            y = y[~np.isnan(y)]
            fig.add_trace(
                go.Scatter(
                    x=x.tolist() + x.tolist()[::-1],
                    y=(y - err).tolist() + (y + err).tolist()[::-1],
                    fill='toself',
                    fillcolor=shape.line.color,
                    opacity=ERROR_AREA_ALPHA,
                    line=dict(color=shape.line.color, width=ERROR_AREA_LINE_WIDTH),
                    hoverinfo='skip',
                    showlegend=False,
                    legendgroup=shape.legendgroup,
                    xaxis=shape.xaxis,
                    yaxis=shape.yaxis,
                    mode='lines'  # no markers
                )
            )

    # plots mean
    if plot_mean:
        if len(data.columns) == 1:
            y_mean = np.nanmean(data.values)  # add horizontal line with mean value
            fig.add_hline(y=y_mean, line_width=2, line_dash='dash', line_color='grey',
                          annotation_text='Mean', annotation_position='bottom right')
        else:
            mean = np.nanmean(data.values, axis=1)  # add mean curve across time
            fig.add_trace(go.Scatter(x=data.index, y=mean, name='Mean',
                                     line=dict(width=2, dash='dash', color='grey')))

    # merge data and errors
    err_data = pd.DataFrame(err_data, index=data.index)
    err_data.columns = [f'{col} error' for col in err_data.columns]
    data_plus_err = pd.concat([data, err_data], axis=1)

    # format and save plot
    if reverse_x_axis:
        fig.update_xaxes(autorange='reversed')
    fig.update_layout(yaxis_range=[y_min, y_max])
    format_and_save_plot(
        fig, data_plus_err, title, output_img, save_csv, False, save_json, width, height, show_legend, show_plot,
        **kwargs)

    return fig


def plot_bar(data: Union[pd.DataFrame, Dict[str, np.ndarray], Dict[str, float]],
             title: Optional[str] = None,
             output_img: Optional[str] = None,
             save_csv: bool = True,
             save_json: bool = True,
             x_label: Optional[str] = None,
             y_label: Optional[str] = None,
             palette: Optional[Union[List[str], str]] = DEF_PALETTE,
             template: str = DEF_TEMPLATE,
             width: Optional[int] = 600,
             height: Optional[int] = 450,
             plot_mean: bool = False,
             plot_error: bool = True,
             error_stat: ErrorStat = ErrorStat.CI95,
             group_by: str = None,
             group_norm: bool = False,
             weights: Optional[str] = None,
             y_min: Optional[float] = None,
             y_max: Optional[float] = None,
             log_y: bool = False,
             orientation: Optional[str] = None,
             show_values: Optional[str] = None,
             show_legend: bool = False,
             show_plot: bool = False,
             **kwargs) -> go.Figure:
    """
    Plots several variables in the same bar chart.
    :param pd.DataFrame data: the object containing the data to be plotted. If a `dict` is given, then each key
    corresponds to the name of a variable, while the values are arrays of shape (steps, ) containing the variables'
    value over time. If a `DataFrame` is provided, then it is assumed each column represents a different variable, while
    the indices represent different timesteps, i.e., corresponding to long-form data.
    :param str title: the title of the plot.
    :param str output_img: the path to the image file in which to save the plot.
    :param bool save_csv: whether to save the (possibly transformed) data in a CSV file.
    :param bool save_json: whether to save the plot data to a Json file for later retrieval/loading.
    :param str x_label: the label of the X axis.
    :param str y_label: the label of the Y axis.
    :param str or list[str] palette: the name of the Plotly palette used to color each series line, or a list containing
    a string representation for the colors to be used. See: https://plotly.com/python/builtin-colorscales/.
    :param str template: the name of the Plotly layout template to be used. Defaults to "plotly_white".
    See: https://plotly.com/python/templates/.
    :param int width: the plot's width.
    :param int height: the plot's height.
    :param bool plot_mean: whether to plot the "mean" series line, corresponding to the average of the series values at
    each discrete step. If only one series is plotted, then an horizontal line with the mean value is added to the plot.
    :param bool plot_error: if the data has more than one sample, then this controls whether the standard error bar is
    plotted in addition to the bar's mean value.
    :param bool error_stat: the statistic used to plot the error bar.
    :param str group_by: controls whether the data should be grouped by some column prior to plotting. This will create
    `N*M` new bar shapes, one for each possible value of the `group_by` column, for each of the other variables/
    columns in the data. The bars will be grouped by the `group_by` variable in the plot.
    :param bool group_norm: If `group_by` is not `None`, then this controls whether the quantities for the other
    variables should be normalized such that the sum for each `group_by` variable value equals to 1. This results in
    a stacked bars plot, each value representing a percentage.
    :param str weights: the data column containing the rows' weights, in which case a weighted average is computed.
    :param float y_min: the minimal value of the y-axis.
    :param float y_max: the maximal value of the y-axis.
    :param bool log_y: whether the y-axis is log-scaled in cartesian coordinates.
    :param str orientation: orientation of the bar chart, one of ['h','v'].
    :param str show_values: whether to show values on each column. If not None, specifies the values' text format.
    :param bool show_legend: whether to show the plot's legend.
    :param bool show_plot: whether to show the plot, in which case a new browser tab would be opened displaying the
    interactive Plotly plot.
    :param kwargs: extra arguments passed to `format_and_save_plot`.
    :rtype: go.Figure
    :return: the Plotly figure created with the given data.
    """
    data = _convert_data(data)  # transform to pandas dataframe first

    if weights is not None:
        assert weights in data.columns, f'Weights column {weights} is not a valid column: {data.columns}'

    # group by given column
    is_grouped = group_by is not None
    if is_grouped:
        # take the mean over groups for each variable
        data, err_data = _get_mean_error_stat(data, error_stat, group_by=group_by, weights=weights, axis=0)

        # checks normalization
        if group_norm:
            norm_factor = (data.sum(axis=1)).values.reshape(-1, 1)
            data /= norm_factor * 0.01
            err_data /= norm_factor * 0.01
            y_label = 'Percentage' if y_label is None else y_label + ' (%)'
    else:
        # take the mean over variables
        data, err_data = _get_mean_error_stat(data, error_stat, weights=weights, axis=0)

    # checks color palette, discretize if needed
    colors = palette
    if isinstance(colors, str) and hasattr(px.colors.qualitative, colors):
        colors = getattr(px.colors.qualitative, colors)  # get discrete color sequence by name
    num_colors = len(data) if (is_grouped and len(data.columns) == 1) else len(data.columns)
    if isinstance(colors, str) or (isinstance(colors, list) and num_colors > len(colors) > 1):
        colors = px.colors.sample_colorscale(palette, np.linspace(0, 1, num_colors))

    # plots data
    fig = px.bar(data if is_grouped else data.T,
                 title=title,
                 labels={'index': x_label or 'Index',
                         'value': y_label or 'Value',
                         'variable': x_label or 'Variable'},
                 barmode='group' if is_grouped and not group_norm else 'relative',
                 text_auto=show_values if show_values is not None else False,
                 color=None if is_grouped else data.columns,
                 color_discrete_sequence=colors,
                 orientation=orientation,
                 log_y=log_y,
                 template=template)

    # updates error bars
    fig.update_traces(dict(error_y=dict(width=ERROR_BAR_WIDTH, thickness=ERROR_BAR_THICKNESS,
                                        visible=bool(plot_error and pd.notna(err_data).sum().sum() > 0))))
    err_data.columns = [str(col) for col in err_data.columns]
    for shape in fig.data:
        shape.update(error_y=dict(array=err_data[shape.name], type='data', symmetric=True))

    # updates bar colors if grouped and single dimension displayed
    if is_grouped and len(data.columns) == 1:
        fig.data[0].marker.color = colors  # apply colors

    # plots mean
    if plot_mean:
        y_mean = np.nanmean(data.values)  # add horizontal line with mean value
        if orientation == 'h':
            fig.add_vline(x=y_mean, line_width=2, line_dash='dash', line_color='grey',
                          annotation_text='Mean', annotation_position='bottom right')
        else:
            fig.add_hline(y=y_mean, line_width=2, line_dash='dash', line_color='grey',
                          annotation_text='Mean', annotation_position='bottom right')

    # merge data and errors
    err_data = pd.DataFrame(err_data, index=data.index)
    err_data.columns = [f'{col} error' for col in err_data.columns]
    data_plus_err = pd.concat([data, err_data], axis=1)

    # format and save plot
    if orientation == 'h':
        fig.update_layout(xaxis_range=[y_min, y_max])
    else:
        fig.update_layout(yaxis_range=[y_min, y_max])

    format_and_save_plot(
        fig, data_plus_err, title, output_img, save_csv, True, save_json, width, height, show_legend, show_plot,
        **kwargs)

    return fig


def plot_histogram(data: Union[pd.DataFrame, Dict[str, np.ndarray]],
                   title: Optional[str] = None,
                   output_img: Optional[str] = None,
                   save_csv: bool = True,
                   save_json: bool = True,
                   x_label: Optional[str] = None,
                   y_label: Optional[str] = None,
                   var_label: Optional[str] = None,
                   palette: Optional[Union[List[str], str]] = DEF_HIST_PALETTE,
                   template: str = DEF_TEMPLATE,
                   width: Optional[int] = 600,
                   height: Optional[int] = 450,
                   plot_mean: bool = False,
                   normalize: bool = False,
                   n_bins: Optional[int] = None,
                   group_by: str = None,
                   x_min: Optional[float] = None,
                   x_max: Optional[float] = None,
                   log_x: bool = False,
                   show_legend: bool = True,
                   show_plot: bool = False,
                   **kwargs) -> go.Figure:
    """
    Plots several variables in the same histogram plot, representing the "distribution" over values.
    :param pd.DataFrame data: the object containing the data to be plotted. If a `dict` is given, then each key
    corresponds to the name of a variable, while the values are arrays of shape (num_samples, ) containing the variables'
    sample values. If a `DataFrame` is provided, then it is assumed each column represents a different variable, while
    the indices represent different samples, i.e., corresponding to long-form data.
    :param str title: the title of the plot.
    :param str output_img: the path to the image file in which to save the plot.
    :param bool save_csv: whether to save the (possibly transformed) data in a CSV file.
    :param bool save_json: whether to save the plot data to a Json file for later retrieval/loading.
    :param str x_label: the label of the X axis.
    :param str y_label: the label of the Y axis.
    :param str var_label: the name given to the group of variables to be plotted.
    :param str or list[str] palette: the name of the Plotly palette used to color each series line, or a list containing
    a string representation for the colors to be used. See: https://plotly.com/python/builtin-colorscales/.
    :param str template: the name of the Plotly layout template to be used. Defaults to "plotly_white".
    See: https://plotly.com/python/templates/.
    :param int width: the plot's width.
    :param int height: the plot's height.
    :param bool plot_mean: whether to plot the "mean" histogram value, corresponding to the average of the series values.
    :param bool normalize: whether the the value of each histogram bin is normalized such that it corresponds to the
    probability that a random event whose distribution is described by the histogram will fall into that bin.
    A negative value of `None` means no length normalization is performed.
    :param int n_bins: the number of histogram bins.
    :param str group_by: controls whether the data should be grouped by some column prior to plotting. This will create
    `N*M` new variables/series, one for each possible value of the `group_by` column, for each of the other variables/
    columns in the data. If `average` is `True`, then the average is performed by original variable across the grouped
    data and not over all variables.
    :param float x_min: the minimal value of the x-axis, i.e., the variables' values.
    :param float x_max: the maximal value of the x-axis, i.e., the variables' values.
    :param bool log_x: whether the y-axis is log-scaled in cartesian coordinates.
    :param bool show_legend: whether to show the plot's legend.
    :param bool show_plot: whether to show the plot, in which case a new browser tab would be opened displaying the
    interactive Plotly plot.
    :param kwargs: extra arguments passed to `format_and_save_plot`.
    :rtype: go.Figure
    :return: the Plotly figure created with the given data.
    """
    data = _convert_data(data)  # transform to pandas dataframe first

    # group by given column
    if group_by is not None and group_by in data.columns:
        dfs = []
        for g, g_df in data.groupby(group_by):
            g_df = g_df.drop(group_by, axis=1)  # drop the group-by column
            g_df = g_df.rename(columns={v: f'{group_by}-{g}-{v}' for v in g_df.columns})
            dfs.append(g_df)
        data = pd.concat(dfs, axis=1)

    # checks color palette, discretize if needed
    colors = palette
    if isinstance(colors, str) and hasattr(px.colors.qualitative, colors):
        colors = getattr(px.colors.qualitative, colors)  # get discrete color sequence by name
    if (isinstance(colors, str) or
            (isinstance(colors, list) and
             (len(data.columns) if isinstance(data, pd.DataFrame) else len(data)) > len(colors) > 1)):
        colors = px.colors.sample_colorscale(colors, np.linspace(0, 1, len(data.columns)))

    # plots data
    fig = px.histogram(data, title=title,
                       labels={'value': x_label or 'Value',
                               'variable': var_label or 'Variable'},

                       nbins=n_bins,
                       range_x=None if x_min is None or x_max is None else [x_min, x_max],
                       log_x=log_x,
                       histnorm='probability density' if normalize else None,
                       color_discrete_sequence=colors,
                       template=template)

    if palette is not None and len(data.columns) == 1:
        # creates a colored histogram based on number of bins
        # see: https://stackoverflow.com/a/66590885/16031961
        f = fig.full_figure_for_development(warn=False)
        x_bins = f.data[0].xbins  # gets plotly-generated bins
        if x_bins.start is not None and x_bins.end is not None and x_bins.size is not None:
            if isinstance(x_bins.start, str):
                # x values are date/times
                if isinstance(x_bins.size, int):
                    base = 'D'
                    delta = np.timedelta64(x_bins.size, 'ms').astype('timedelta64[D]')
                else:
                    base = x_bins.size[0]
                    delta = np.timedelta64(int(x_bins.size[1:]), base)
                x_vals = np.arange(np.datetime64(x_bins.start).astype(f'datetime64[{base}]'),
                                   np.datetime64(x_bins.end).astype(f'datetime64[{base}]') + delta,
                                   delta)
            else:
                x_vals = np.arange(x_bins.start, x_bins.end + x_bins.size, x_bins.size)
            n_bins = len(x_vals)

        if n_bins is not None:
            # checks color palette, discretize if needed
            colors = palette
            if isinstance(colors, str) and hasattr(px.colors.qualitative, colors):
                colors = getattr(px.colors.qualitative, colors)  # get discrete color sequence by name
            if (isinstance(colors, str) or
                    (isinstance(colors, list) and n_bins != len(colors))):
                colors = px.colors.sample_colorscale(colors, np.linspace(0, 1, n_bins))

            fig.data[0].marker.color = colors  # apply colors
            fig.update_layout(bargap=0, bargroupgap=0.0)

    # plots mean
    if plot_mean:
        y_mean = np.nanmean(data.values)  # add horizontal line with mean value
        fig.add_vline(x=y_mean, line_width=2, line_dash='dash', line_color='grey',
                      annotation_text='Mean', annotation_position='top right')

    # adjusts elements
    if x_label is not None:
        fig.layout.xaxis.title.update(text=x_label)
    if y_label is not None:
        fig.layout.yaxis.title.update(text=y_label)

    # format and save plot
    format_and_save_plot(
        fig, data, title, output_img, save_csv, False, save_json, width, height, show_legend, show_plot,
        **kwargs)

    return fig


def plot_matrix(data: Union[pd.DataFrame, np.ndarray],
                title: Optional[str] = None,
                output_img: Optional[str] = None,
                save_csv: bool = True,
                save_json: bool = True,
                x_label: Optional[str] = None,
                y_label: Optional[str] = None,
                var_label: Optional[str] = None,
                palette: Optional[Union[List[str], str]] = DEF_MATRIX_PALETTE,
                template: str = DEF_TEMPLATE,
                width: Optional[int] = 600,
                height: Optional[int] = 450,
                normalize: bool = False,
                show_scale: bool = True,
                show_values: bool = True,
                z_min: Optional[float] = None,
                z_max: Optional[float] = None,
                symmetrical: bool = False,
                interpolate: bool = False,
                nan_color: Optional[str] = 'black',
                show_plot: bool = False,
                **kwargs) -> go.Figure:
    """
    Plots a given matrix of values in a heatmap plot.
    :param pd.DataFrame data: the object containing the data to be plotted. If a `numpy.array` is given, then it is
    converted to a `DataFrame` with generic names for the x and y data labels. If a `DataFrame` is provided, then the
    x and y data labels are given by the column and index names, respectively.
    :param str title: the title of the plot.
    :param str output_img: the path to the image file in which to save the plot.
    :param bool save_csv: whether to save the (possibly transformed) data in a CSV file.
    :param bool save_json: whether to save the plot data to a Json file for later retrieval/loading.
    :param str x_label: the label of the X axis.
    :param str y_label: the label of the Y axis.
    :param str var_label: the name given to the variable to be plotted, whose values define the color scale.
    :param str or list[str] palette: the name of the Plotly palette used to color each series line, or a list containing
    a string representation for the colors to be used. See: https://plotly.com/python/builtin-colorscales/.
    :param str template: the name of the Plotly layout template to be used. Defaults to "plotly_white".
    See: https://plotly.com/python/templates/.
    :param int width: the plot's width.
    :param int height: the plot's height.
    :param bool normalize: whether to normalize data using min-max scaling.
    :param bool show_scale: whether to show the color scale on the right of the plot.
    :param bool show_values: whether to show the matrix entries' values.
    :param float z_min: the minimum value of data that defines the color scale.
    :param float z_max: the maximum value of data that defines the color scale.
    :param bool symmetrical: whether the given data represents a symmetrical, square matrix, i.e., where
    `data[i,j]==data[j,i]`. If `True`, then only the bottom half will be displayed.
    :param bool interpolate: whether to interpolate between the matrix values to get a smoothed image.
    :param str nan_color: the color used to fill in blank/nan entries in the matrix.
    :param bool show_plot: whether to show the plot, in which case a new browser tab would be opened displaying the
    interactive Plotly plot.
    :param kwargs: extra arguments passed to `format_and_save_plot`.
    :rtype: go.Figure
    :return: the Plotly figure created with the given data.
    """
    # transform to pandas dataframe first
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    # normalize data
    if normalize:
        data = data.astype(float, copy=True)
        min_val = data.min().min()
        max_val = data.max().max()
        norm_values = (data.values - min_val) / (max_val - min_val)  # normalize to [0-1]
        if z_min is not None and z_max is not None:
            norm_values = z_min + norm_values * (z_max - z_min)  # normalize to given range
        data.values[:] = norm_values

    # checks symmetrical matrix
    values_text = None
    if symmetrical and data.shape[0] == data.shape[1]:
        idxs = np.indices((data.shape[0], data.shape[1]))
        data = data.copy()
        data.values[np.where(idxs[1] > idxs[0])] = np.nan  # "remove" value for upper right coordinates

        # hide top and right grid outline
        kwargs.update(dict(xaxis=dict(mirror=False), yaxis=dict(mirror=False)))

        if show_values:
            # trick to hide values for upper right coordinates, otherwise will show as "0.00"
            values_text = data.apply(lambda s: s.apply(lambda x: f'{x:.2f}')).values
            values_text[np.where(idxs[1] > idxs[0])] = ''

    # plots data
    fig = px.imshow(data,
                    title=title,
                    labels=dict(x=x_label, y=y_label, color=var_label),
                    zmin=z_min, zmax=z_max,
                    color_continuous_scale=palette,
                    template=template,
                    text_auto=False if not show_values else True if 'int' in str(data.dtypes[0]) else '.2f')

    if values_text is not None:
        fig.update_traces(text=values_text, texttemplate='%{text}')  # do not use text_auto, set labels manually

    # show color bar and hide grid lines
    fig.update_layout(coloraxis_showscale=show_scale,
                      plot_bgcolor=nan_color,
                      coloraxis_colorbar=dict(outlinecolor='black', outlinewidth=1, title=None),
                      xaxis_showgrid=False, yaxis_showgrid=False,
                      xaxis_zeroline=False, yaxis_zeroline=False)

    # interpolation
    fig.data[0].zsmooth = 'best' if interpolate else False

    # format and save plot
    format_and_save_plot(
        fig, data, title, output_img, save_csv, True, save_json, width, height, False, show_plot,
        **kwargs)

    return fig


def plot_radar(data: Union[pd.DataFrame, Dict[str, np.ndarray], Dict[str, float]],
               title: Optional[str] = None,
               output_img: Optional[str] = None,
               save_csv: bool = True,
               save_json: bool = True,
               var_label: Optional[str] = None,
               value_label: Optional[str] = None,
               palette: Optional[Union[List[str], str]] = DEF_PALETTE,
               template: str = DEF_TEMPLATE,
               width: Optional[int] = 600,
               height: Optional[int] = 450,
               plot_mean: bool = False,
               group_by: str = None,
               min_val: Optional[float] = None,
               max_val: Optional[float] = None,
               alpha: float = RADAR_ALPHA,
               show_legend: bool = False,
               show_plot: bool = False,
               **kwargs) -> go.Figure:
    """
    Plots several variables in the same radar (spider, star) chart. Each data column corresponds to a polar axis in the
    plot, while the value is determined by averaging each series across the rows. To plot multiple traces in the same
    plot, we can group the data by one of the columns.
    :param pd.DataFrame data: the object containing the data to be plotted. If a `dict` is given, then each key
    corresponds to the name of a variable, while the values are arrays of shape (steps, ) containing the variables'
    value over time. If a `DataFrame` is provided, then it is assumed each column represents a different variable, while
    the indices represent different timesteps, i.e., corresponding to long-form data.
    :param str title: the title of the plot.
    :param str output_img: the path to the image file in which to save the plot.
    :param bool save_csv: whether to save the (possibly transformed) data in a CSV file.
    :param bool save_json: whether to save the plot data to a Json file for later retrieval/loading.
    :param str var_label: the name given to the group of variables to be plotted (one per polar axis).
    :param str value_label: the label of the quantities associated with each variable (the polar coordinates).
    :param str or list[str] palette: the name of the Plotly palette used to color each series line, or a list containing
    a string representation for the colors to be used. See: https://plotly.com/python/builtin-colorscales/.
    :param str template: the name of the Plotly layout template to be used. Defaults to "plotly_white".
    See: https://plotly.com/python/templates/.
    :param int width: the plot's width.
    :param int height: the plot's height.
    :param bool plot_mean: whether to plot the "mean" trace, corresponding to the average of the groups' values.
    :param str group_by: controls whether the data should be grouped by some column prior to plotting. This will create
    `N` new traces on the plot, one for each possible value of the `group_by` column.
    :param float min_val: the minimal value of the polar axes.
    :param float max_val: the maximal value of the polar axes.
    :param float alpha: the transparency (alpha channel) value of radar shapes.
    :param bool show_legend: whether to show the plot's legend.
    :param bool show_plot: whether to show the plot, in which case a new browser tab would be opened displaying the
    interactive Plotly plot.
    :param kwargs: extra arguments passed to `format_and_save_plot`.
    :rtype: go.Figure
    :return: the Plotly figure created with the given data.
    """
    data = _convert_data(data)  # transform to pandas dataframe first

    # group by given column
    is_grouped = group_by is not None and group_by in data.columns

    # take the mean over groups for each variable (is_grouped) or the mean over variables (not is_grouped)
    data, _ = _get_mean_error_stat(data, ErrorStat.StdDev, group_by=group_by if is_grouped else None, axis=0)

    if isinstance(data, pd.Series):
        data = data.to_frame().T  # get dataframe

    # checks color palette, discretize if needed
    colors = palette
    if isinstance(colors, str) and hasattr(px.colors.qualitative, colors):
        colors = getattr(px.colors.qualitative, colors)  # get discrete color sequence by name
    if isinstance(colors, str) or (isinstance(colors, list) and len(data) > len(colors) > 1):
        colors = px.colors.sample_colorscale(colors, np.linspace(0, 1, len(data)))

    # convert all colors to rgb format to allows playing with alpha
    colors = px.colors.convert_colors_to_same_type(colors, colortype='rgb')[0]

    # plots data, one scatter polar plot per row
    fig = go.Figure()
    hover_template_base = f'{"theta" if var_label is None else var_label}: %{{theta}}<br>' + \
                          f'{"r" if value_label is None else value_label}: %{{r}}'
    thetas = list(data.columns) + [data.columns[0]]  # add first point to close final segment
    for i, (idx, row) in enumerate(data.iterrows()):
        color = colors[i % len(colors)] if colors is not None else None
        fill_color = None if color is None else f'rgba{color[3:-1]}, {alpha})'  # add custom alpha
        r = list(row.values) + [row.values[0]]  # add first point to close final segment
        hover_template = hover_template_base + (f'<br>{group_by}: {idx}' if is_grouped else '')
        fig.add_trace(go.Scatterpolar(
            theta=thetas, r=r, hovertemplate=hover_template,
            line=dict(width=1, color=color),
            marker=dict(symbol=i, size=MARKER_SIZE),
            fillcolor=fill_color, fill='toself', name=str(idx),
        ))

    # plots mean
    if plot_mean and len(data) > 1:
        vals_mean = list(np.nanmean(data.values, axis=0))  # add line with mean values
        vals_mean += [vals_mean[0]]  # add first point to close final segment
        fig.add_trace(go.Scatterpolar(
            theta=thetas, r=vals_mean, hovertemplate=hover_template_base,
            line=dict(width=2, dash='dash', color='grey'),
            marker=dict(symbol=len(data), size=MARKER_SIZE), name='Mean'
        ))

    # format layout
    fig.update_layout(
        title_text=title,
        template=template,
        legend_title=group_by,
        polar=dict(
            radialaxis=dict(
                visible=True,
                showline=False,
                tickfont_size=AXES_TITLE_FONT_SIZE - 2,
                range=[min_val, max_val] if min_val is not None and max_val is not None else None
            ),
            angularaxis=dict(gridcolor='darkgrey',
                             linecolor='black',
                             tickfont_size=AXES_TITLE_FONT_SIZE)
        ),
    )

    # format and save plot
    format_and_save_plot(fig, data, title, output_img, save_csv, True, save_json,
                         width, height, show_legend, show_plot,
                         margin=dict(l=0, r=0, b=20, t=35), **kwargs)

    return fig


def plot_embeddings(data: Union[pd.DataFrame, Dict[str, np.ndarray]],
                    label_col: str,
                    title: Optional[str] = None,
                    output_img: Optional[str] = None,
                    save_csv: bool = True,
                    save_json: bool = True,
                    palette: Optional[Union[List[str], str]] = DEF_PALETTE,
                    template: str = DEF_TEMPLATE,
                    width: Optional[int] = 600,
                    height: Optional[int] = 450,
                    dims: int = 2,
                    seed: int = 17,
                    show_legend: bool = True,
                    show_plot: bool = False,
                    **kwargs) -> go.Figure:
    """
    Performs dimensionality reduction over the given set of numeric feature vectors, considered as embeddings, and
    plots a 2D or 3D scatter plot from the resulting projection colored by the corresponding class labels.
    Note: requires the `scikit-learn` package to perform t-SNE dimensionality reduction.
    :param pd.DataFrame data: the object containing the data to be plotted. If a `dict` is given, then each key
    corresponds to the name of an embedding variable/feature, and values arrays of shape (n_embed, ) with the values
    for each instance/datapoint.
    :param str label_col: the name of the column containing the labels of each embedding/vector for color plotting.
    :param str title: the title of the plot.
    :param str output_img: the path to the image file in which to save the plot.
    :param bool save_csv: whether to save the (possibly transformed) data in a CSV file.
    :param bool save_json: whether to save the plot data to a Json file for later retrieval/loading.
    :param str or list[str] palette: the name of the Plotly palette used to color each series line, or a list containing
    a string representation for the colors to be used. See: https://plotly.com/python/builtin-colorscales/.
    :param str template: the name of the Plotly layout template to be used. Defaults to "plotly_white".
    See: https://plotly.com/python/templates/.
    :param int width: the plot's width.
    :param int height: the plot's height.
    :param int dims: the number of dimensions of the resulting projection space (either 2 or 3).
    :param int seed: the seed used by the t-SNE algorithm.
    :param bool show_legend: whether to show the plot's legend.
    :param bool show_plot: whether to show the plot, in which case a new browser tab would be opened displaying the
    interactive Plotly plot.
    :param kwargs: extra arguments passed to `format_and_save_plot`.
    :rtype: go.Figure
    :return: the Plotly figure created with the given data.
    """
    data = _convert_data(data)  # transform to pandas dataframe first

    # perform dimensionality reduction on the features/embedding
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=dims, random_state=seed)
    projections = tsne.fit_transform(data.drop(label_col, axis=1).values)

    # map labels to colors (discretize if needed)
    labels = sorted(data[label_col].unique())
    colors = palette
    if isinstance(colors, str) and hasattr(px.colors.qualitative, colors):
        colors = getattr(px.colors.qualitative, colors)  # get discrete color sequence by name
    if isinstance(colors, str) or (isinstance(colors, list) and len(labels) > len(colors) + 1 > 1):
        colors = px.colors.sample_colorscale(colors, np.linspace(0, 1, len(labels)))
    colors = {label: colors[i] for i, label in enumerate(labels)}

    if dims == 2:
        fig = px.scatter(
            projections, x=0, y=1,
            title=title,
            color=data[label_col].values, labels={'color': label_col},
            color_discrete_map=colors,
            template=template,
        )
        fig.update_layout(xaxis_title='Dim 0', yaxis_title='Dim 1')
    elif dims == 3:
        fig = px.scatter_3d(
            projections, x=0, y=1, z=2,
            title=title,
            color=data[label_col], labels={'color': label_col},
            color_discrete_map=colors,
            template=template,
        )
        fig.update_layout(scene=dict(xaxis_title='Dim 0', yaxis_title='Dim 1', zaxis_title='Dim 2'))
    else:
        raise ValueError(f'Only visualization up to 3 dimensions supported, but {dims} requested')

    # format and save plot
    format_and_save_plot(
        fig, data, title, output_img, save_csv, False, save_json, width, height, show_legend, show_plot,
        **kwargs)

    return fig


def format_and_save_plot(fig: go.Figure,
                         data: Optional[pd.DataFrame] = None,
                         title: Optional[str] = None,
                         output_img: Optional[str] = None,
                         save_csv: bool = True,
                         save_index: bool = False,
                         save_json: bool = True,
                         width: Optional[int] = 600,
                         height: Optional[int] = 450,
                         show_legend: bool = True,
                         show_plot: bool = False,
                         margin: Optional[Dict[str, float]] = None,
                         font_size: Optional[int] = ALL_FONT_SIZE,
                         title_font_size: Optional[int] = TITLE_FONT_SIZE,
                         axes_title_font_size: Optional[int] = AXES_TITLE_FONT_SIZE,
                         **layout_kwargs):
    """
    Formats the given plot according to the common "look-and-feel" of the library, and optionally saves the figure to
    files with different formats.
    :param go.Figure fig: the figure to be plotted and saved.
    :param pd.DataFrame data: the original data to be saved to file.
    :param str title: the title of the plot.
    :param str output_img: the path to the image file in which to save the plot.
    :param bool save_csv: whether to save the (possibly transformed) data in a CSV file.
    :param bool save_index: whether to save the dataframe's index in the CSV file.
    :param bool save_json: whether to save the plot data to a Json file for later retrieval/loading.
    :param int width: the plot's width.
    :param int height: the plot's height.
    :param bool show_legend: whether to show the plot's legend.
    :param bool show_plot: whether to show the plot, in which case a new browser tab would be opened displaying the
    interactive Plotly plot.
    :param dict margin: the figure's margin parameters (l, r, t, b).
    :param int font_size: the default font size for all figure elements.
    :param int title_font_size: the font size of the figure title.
    :param int axes_title_font_size: the font size of the axes' titles.
    :param layout_kwargs: the extra arguments to pass to `update_layout` of the figure.
    """
    # saves to csv
    if data is not None and save_csv and output_img is not None:
        data.to_csv(get_file_changed_extension(output_img, 'csv'), index=save_index)

    # saves to json
    if save_json and output_img is not None:
        with open(get_file_changed_extension(output_img, 'json'), 'w') as fp:
            fp.write(fig.to_json())

    # tweaks layout
    axis_layout = dict(mirror=True, ticks='outside', showline=True, linecolor='black', title=dict(standoff=10))
    margin = margin if margin is not None else dict(l=0, r=0, b=0, t=title_font_size + 12 if title else 0)
    fig.update_layout(showlegend=show_legend, legend_itemclick='toggle',
                      legend=dict(bordercolor='black', borderwidth=1),
                      xaxis=axis_layout, yaxis=axis_layout, margin=margin,
                      width=width, height=height)

    fig.update_layout(**layout_kwargs)  # update layout with custom params separately

    if show_plot:
        fig.show()

    # saves to image file
    if output_img is not None:
        fig = go.Figure(fig)  # makes copy to change fonts only for printing
        fig.update_layout(
            font_size=font_size,
            title_font_size=title_font_size,
            xaxis_title_font_size=axes_title_font_size,
            yaxis_title_font_size=axes_title_font_size)
        fig.write_image(output_img)


def dummy_plotly(sleep: float = 0.5):
    """
    Utility method to generate and print a plotly plot to file, otherwise in the first plot, a message appears in the
    bottom-right portion of the image stating that a javascript library is loading.
    See: https://github.com/plotly/plotly.py/issues/3469.
    :param float sleep: the time, in seconds, to wait after the dummy plot has been produced, to make sure the libraries
    are loaded such that the message does not appear in the next generated image.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'tmpfile.pdf')
        x = np.arange(10)
        fig = go.Figure(data=go.Scatter(x=x, y=x ** 2))
        fig.write_image(path)
        import time
        time.sleep(sleep)  # not sure why, but waiting helps the message be absent in subsequent plots..


def distinct_colors(n: int, array: bool = False) -> Union[List[str], np.ndarray]:
    """
    Generates N visually-distinct colors.
    :param int n: the number of colors to generate.
    :param bool array: whether to return a normalized array instead of a plotly string.
    :rtype: list[str] or np.ndarray
    :return: a list of plotly colors in the rgb(R, G, B) format.
    """
    colors = [colorsys.hls_to_rgb(i / n, .65, .9) for i in range(n)]
    if array:
        return np.array(colors)

    return ['rgb(' + ','.join([str(int(255 * c)) for c in color]) + ')' for color in colors]
