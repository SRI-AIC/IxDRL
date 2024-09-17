import logging
import os
import threading
from typing import List, Tuple, Optional, Dict, NamedTuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from plotly.io import from_json

from ixdrl import Rollouts
from ixdrl.analysis import RolloutsAnalyses, ROLLOUT_ID_COL, TIMESTEP_COL
from ixdrl.bin.analyze import INTERESTINGNESS_DATA_FILE, INTERESTINGNESS_PLOTS_FILE, \
    INTERESTINGNESS_PANDAS_FILE
from ixdrl.data_collection import INTERACTION_DATA_FILE, INTERACTION_PLOTS_FILE, MEAN_ROLLOUT_ID
from ixdrl.types import get_mean_data
from ixdrl.util.cmd_line import ErrorMsgArgumentParser
from ixdrl.util.io import load_object
from ixdrl.util.plot import plot_timeseries
from ixdrl.util.streamlit import temp_widget, create_video_player, format_markdown
from ixdrl.util.video import get_video_metadata

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

INFO_TIMEOUT = 3
PAGE_TITLE = 'RL Inspection App'
PAGE_ICON = '🧪'
LAYOUT = 'wide'  # 'centered'  # 'wide'
INITIAL_SIDEBAR_STATE = 'expanded'
INITIAL_VIDEO_HEIGHT = 340
VIDEO_PLAYER_ID = 'inspect-video-player'

CUR_STEP_LINE_WIDTH = 3
SELECTED_STEP_COLOR = 'gold'
STEP_RANGE_OPACITY = 0.2
DATA_HIGHLIGHT_COLOR = 'gold'


def _customize_ui():
    # initial config
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state=INITIAL_SIDEBAR_STATE,
    )

    # https://towardsdatascience.com/5-ways-to-customise-your-streamlit-ui-e914e458a17c
    st.markdown(f""" <style>
        .reportview-container .main .block-container{{
            padding-top: {0}rem;
            padding-right: {2}rem;
            padding-left: {2}rem;
            padding-bottom: {0}rem;
        }} </style> """, unsafe_allow_html=True)


class _Data(NamedTuple):
    interaction_data: Rollouts
    interaction_plots: Dict[str, Dict[str, str]]
    interestingness: RolloutsAnalyses
    interestingness_dims: List[str]
    interestingness_plots: Dict[str, Dict[str, str]]
    interestingness_df: pd.DataFrame
    rollout_ids: List[str]
    training_df: Optional[pd.DataFrame]


@st.cache(allow_output_mutation=True)
def _load_data(temps: List[threading.Thread]) -> Optional[_Data]:
    # list of files to be loaded and corresponding params
    data_files = {
        'interaction_data': os.path.join(args.interaction, INTERACTION_DATA_FILE),
        'interaction_plots': os.path.join(args.interaction, INTERACTION_PLOTS_FILE),
        'interestingness': os.path.join(args.interestingness, INTERESTINGNESS_DATA_FILE),
        'interestingness_plots': os.path.join(args.interestingness, INTERESTINGNESS_PLOTS_FILE),
        'interestingness_df': os.path.join(args.interestingness, INTERESTINGNESS_PANDAS_FILE),
    }

    # checks and loads data from files
    data = {}
    for param, data_file in data_files.items():
        if not os.path.isfile(data_file):
            st.error(f'Could not find {param.replace("_", " ")} file: {data_file}')
            return None
        with st.spinner(f'Loading data from {data_file}...'):
            data[param] = load_object(data_file)

    if args.training is not None and os.path.isfile(args.training):
        with st.spinner(f'Loading data from {args.training}...'):
            data['training_df'] = pd.read_csv(args.training)
    else:
        data['training_df'] = None
        msg = f'Could not find training file: {args.training}'
        temps.append(temp_widget('warning', INFO_TIMEOUT, msg))
        logging.warning(msg)

    # check data consistency
    rollout_ids = list(data['interaction_data'].keys())
    if set(rollout_ids).isdisjoint(list(data['interestingness'].values())[0].keys()):
        st.error('Rollout ids do not match between interaction and interestingness data')
        return None

    # prints info to UI and console/log
    msg = f'Loaded data for {len(rollout_ids)} rollouts'
    temps.append(temp_widget('info', INFO_TIMEOUT, msg))
    logging.info(msg)

    interestingness_dims = list(data['interestingness'].keys())

    return _Data(rollout_ids=rollout_ids, interestingness_dims=interestingness_dims, **data)


def _sort_rollouts(data: _Data, sort_by: str, sort_order: str) -> List[str]:
    if sort_by == 'ID':
        sort_func = lambda rollout_id: rollout_id
    elif sort_by == 'Length':
        sort_func = lambda rollout_id: np.nanmax(data.interaction_data[rollout_id].data.timesteps)
    elif sort_by == 'Total reward':
        sort_func = lambda rollout_id: np.nansum(data.interaction_data[rollout_id].data.reward)
    elif sort_by == 'Discounted reward':
        sort_func = lambda rollout_id: np.nansum(np.power(data.interaction_data[rollout_id].discount,
                                                          data.interaction_data[rollout_id].data.timesteps) *
                                                 data.interaction_data[rollout_id].data.reward[:, 0])
    elif sort_by in data.interestingness_dims:
        sort_func = lambda rollout_id: np.nanmean(data.interestingness[sort_by][rollout_id])
    else:
        raise NotImplementedError(f'Unknown sorting function: {sort_by}')
    sorted_idxs = np.argsort([sort_func(rollout_id) for rollout_id in data.rollout_ids])
    if sort_order == 'Descending':
        sorted_idxs = sorted_idxs[::-1]
    return np.array(data.rollout_ids)[sorted_idxs].tolist()


def _print_rollout_info(rollout_id: str, data: _Data):
    datapoint = data.interaction_data[rollout_id]
    int_data = datapoint.data
    timesteps = np.array(int_data.timesteps)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('**Metadata:**')
        st.markdown(format_markdown({
            'ID': rollout_id,
            'Task': datapoint.env_id,
            'Length': np.max(timesteps),
            'Discount factor': datapoint.discount}))

    with c2:
        st.markdown('**Avg. Interestingness:**')
        st.markdown(format_markdown({
            dim: np.mean(data.interestingness[dim][rollout_id]) for dim in data.interestingness_dims
        }))

    with c3:
        st.markdown('**Interaction data:**')
        st.markdown(format_markdown({
            # 'Action dimensions': datapoint.action_dims,
            'Total reward': np.sum(int_data.reward),
            'Discounted reward': np.sum(np.power(datapoint.discount, timesteps) * int_data.reward[:, 0]),
            'Mean reward': (np.mean(int_data.reward[:, 0]), np.std(int_data.reward[:, 0])),
            'Mean value': (np.nanmean(get_mean_data(int_data.value)),
                           np.nanstd(get_mean_data(int_data.value))),
            'Mean action probability': (np.nanmean(get_mean_data(int_data.action_prob)),
                                        np.nanstd(get_mean_data(int_data.action_prob)))
        }))


def _add_plots_section(section_name: str,
                       plots: Dict[str, str],
                       step_range: Optional[Tuple[int, int]] = None,
                       cur_step: Optional[int] = None,
                       expanded: bool = False,
                       use_container_width: bool = True):
    # adds a plot selector in an expander
    with st.expander(section_name, expanded=expanded):
        # gets stored figure (copy) and change elements
        option = st.selectbox('Statistic:', list(plots.keys()))  # add selection
        fig = from_json(plots[option])
        fig.layout.title.text = None  # remove title, not needed
        if isinstance(fig.data[0], (go.Scatter, go.Scattergl)):
            # assume time series, fix step since interaction data does not contain first two steps
            x = fig.data[0].x[cur_step]  # gets timestep corresponding to frame idx
            fig.add_vline(x=x, line_width=CUR_STEP_LINE_WIDTH, line_dash='dash', line_color=SELECTED_STEP_COLOR,
                          annotation_font_color=SELECTED_STEP_COLOR, annotation_text='Current timestep',
                          annotation_position='top right')  # add vertical line

            # add selected area rectangle whenever sub-range selected
            x0 = fig.data[0].x[step_range[0]]  # gets timesteps corresponding to frame idx
            x1 = fig.data[0].x[step_range[1]]
            if x0 > 1 or x1 < np.max(fig.data[0].x):
                fig.add_vrect(x0=x0, x1=x1, fillcolor=SELECTED_STEP_COLOR, opacity=STEP_RANGE_OPACITY, line_width=0)

            for t in fig.data:
                if isinstance(t, go.Scatter):
                    t.mode = 'lines+markers'
                    t.marker.opacity = 0
        # fig.layout.paper_bgcolor = 'rgba(0,0,0,0)'
        # fig.layout.plot_bgcolor = 'rgba(0,0,0,0)'

        st.plotly_chart(fig, use_container_width=use_container_width)  # add plot
        # selected_points = plotly_events(fig, click_event=True, select_event=True)


def _add_rollout_table_section(rollout_id: str, data: _Data):
    with st.expander('Rollout Data', expanded=False):

        # select rollout data
        df: pd.DataFrame = data.interestingness_df[data.interestingness_df[ROLLOUT_ID_COL] == rollout_id]
        df = df.loc[:, df.columns != ROLLOUT_ID_COL]
        df.set_index(TIMESTEP_COL, inplace=True)

        # let user select highlight
        option = st.selectbox('Highlight:', ['None', 'Min', 'Max', 'Equals 0', 'Greater than 0', 'Less than 0'])
        df = df.astype(np.float64)
        style = df.style.format('{:0.2f}')
        if option == 'Min':
            style = style.highlight_min(color=DATA_HIGHLIGHT_COLOR, axis=0)
        elif option == 'Max':
            style = style.highlight_max(color=DATA_HIGHLIGHT_COLOR, axis=0)
        elif option == 'Equals 0':
            style = style.highlight_between(color=DATA_HIGHLIGHT_COLOR, left=0, right=0, axis=0)
        elif option == 'Greater than 0':
            style = style.highlight_between(color=DATA_HIGHLIGHT_COLOR, left=0, right=1, inclusive='right', axis=0)
        elif option == 'Less than 0':
            style = style.highlight_between(color=DATA_HIGHLIGHT_COLOR, left=-1, right=0, inclusive='left', axis=0)

        st.markdown('<style> .data .col_heading, .row_heading {text-align: right !important} </style>',
                    unsafe_allow_html=True)
        st.table(style)


def _add_training_plots(data: _Data):
    if data.training_df is None:
        return

    with st.expander('Training Information', expanded=False):
        data.training_df.columns = [col.title().replace('_', ' ') for col in data.training_df.columns]
        option = st.selectbox('Statistic:', data.training_df.columns)
        data = data.training_df[option]
        fig = plot_timeseries(data.to_frame(), show_legend=False, x_label='Iteration')
        st.plotly_chart(fig, use_container_width=True)


def app():
    _customize_ui()

    # st.header('Interestingness Visualizer')

    # loads and checks data
    temps = []
    data = _load_data(temps)
    if data is None:
        return

    # adds rollout selection options to sidebar
    st.sidebar.subheader('Options')
    with st.sidebar.expander('Rollout Selection', expanded=True):
        sort_by = st.selectbox(
            'Sort by:', ['ID', 'Length', *data.interestingness_dims, 'Total reward', 'Discounted reward'])
        sort_order = st.radio('Sort order:', ['Ascending', 'Descending'])
        rollout_ids = _sort_rollouts(data, sort_by, sort_order)
        rollout_id = st.selectbox(f'Select Rollout (total {len(data.interaction_data)}):', rollout_ids)

    # checks replay video file
    video_file = data.interaction_data[rollout_id].video_file
    video_file = video_file if os.path.isfile(video_file) else os.path.join(args.video, video_file)
    if not os.path.isfile(video_file):
        st.error(f'Could not find video file for rollout "{rollout_id}": {video_file}')
        return

    # adds video options to sidebar
    metadata = get_video_metadata(video_file)
    with st.sidebar.expander('Video Player', expanded=True):
        height = INITIAL_VIDEO_HEIGHT / metadata.height  # aim for a fixed height initially
        video_height = st.slider('Height:', 0.0, float(np.ceil(height)), height)
        playback_rate = st.slider('Playback Rate:', 0.1, 4.0, 0.5)

    # adds main elements

    # adds video player elements
    video_container = st.container()
    frame_range_container = st.empty()
    cur_frame_container = st.empty()

    c1, c2 = st.columns([5, 95])

    with c1:
        play_button_container = c1.empty()
    with c2:
        reset_button_container = c2.empty()
    step_range, cur_step = create_video_player(
        video_file,
        video_container, cur_frame_container, frame_range_container, play_button_container, reset_button_container,
        height=video_height,
        playback_rate=playback_rate,
        player_id=VIDEO_PLAYER_ID,
        min_frame=0,
        max_frame=metadata.total_frames - 2,  # videos contain one extra frame (the last) for which there's no data
        cur_frame_label='Current timestep:',
        frame_range_label='Timestep range:')

    st.markdown('---')

    # adds stats and plots
    with st.expander('Rollout Info', expanded=False):
        _print_rollout_info(rollout_id, data)

    rollout_plots = {**data.interestingness_plots[rollout_id], **data.interaction_plots[rollout_id]}
    _add_plots_section('Rollout-Specific Plots', rollout_plots, step_range, cur_step)
    mean_plots = {**data.interestingness_plots[MEAN_ROLLOUT_ID], **data.interaction_plots[MEAN_ROLLOUT_ID]}
    _add_plots_section('Mean Rollout Plots', mean_plots, step_range, cur_step)

    _add_rollout_table_section(rollout_id, data)

    _add_training_plots(data)

    # finally, wait for all temporary widgets to close
    for t in temps:
        t.join()


if __name__ == '__main__':

    # create parser and parse args
    parser = ErrorMsgArgumentParser()
    parser.add_argument('--interaction', '-d', type=str, default=None, required=True,
                        help='The directory containing the interaction data for a series of rollouts. '
                             'The data should have been produced by any of the `bin.collect.*` scripts.')
    parser.add_argument('--interestingness', '-i', type=str, default=None, required=True,
                        help='The directory containing the interestingness analyses results for a series of '
                             'rollouts. The data should have been produced by the `bin.analyze` script.')
    parser.add_argument('--training', '-t', type=str, default=None,
                        help='The path to a CSV file containing information about the agent\'s training progress.')
    parser.add_argument('--video', '-v', type=str, default='',
                        help='The path to the root directory of video files defined in the rollouts data.')
    args = parser.parse_args()

    # check parsing error
    if parser.error_msg is not None:
        st.error(f'Could not start the app: {parser.error_msg}')
        st.markdown(f'```{parser.format_help()}```')
    else:
        app()
