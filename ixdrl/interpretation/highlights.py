import logging
import os
from typing import Dict, Tuple, List, Optional

import pandas as pd
import plotly.graph_objs as go
import skvideo.io
import tqdm
from PIL import Image
from plotly.io import from_json

from ixdrl.analysis import ROLLOUT_ID_COL, TIMESTEP_COL
from ixdrl.interpretation import get_clean_filename
from ixdrl.util.io import create_clear_dir
from ixdrl.util.mp import run_parallel
from ixdrl.util.plot import format_and_save_plot, dummy_plotly
from ixdrl.util.video import fade_video, save_video

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

DIMENSION_COL = 'Dimension'
VALUE_COL = 'Value'
LABEL_COL = 'Label'
ROLLOUT_LEN_COL = 'Rollout Length'


def extract_highlights(df: pd.DataFrame,
                       dimensions: List[str],
                       output_dir: str,
                       figures: Dict[str, Dict[str, str]],
                       rollout_meta: Dict,
                       max_per_dim: int = 10,
                       iqr_mul: float = 1.5,
                       record_timesteps: int = 41,
                       fade_ratio: float = 0.25,
                       processes: int = -1,
                       img_format: str = 'pdf',
                       videos_dir: Optional[str] = None):
    """
    Captures short video clips highlighting important moments of the agent's interaction with the environment as
    dictated by the interestingness analyses. The goal is to summarize the agent's aptitude in the task, both in
    terms of its capabilities and limitations according to different criteria.
    The key moments are selected for each dimension using IQR outlier detection.
    See [1]; inspired by the approach in [2].
    [1] - Sequeira, P., & Gervasio, M. (2020). Interestingness elements for explainable reinforcement learning:
    Understanding agents' capabilities and limitations. Artificial Intelligence, 288, 103367.
    [2] - Amir, D., & Amir, O. (2018). Highlights: Summarizing agent behavior to people. In AAMAS 2018.
    :param pd.DataFrame df: the interestingness dataset (interestingness dimension per timestpe of each rollout)
    from which to extract highlights.
    :param list[str] dimensions: the list of dimensions for which highlights are going to be extracted.
    :param str output_dir: the path to the directory in which to save highlights.
    :param dict[str, dict[str, str]] figures: the dictionary containing the figures for each rollout.
    :param dict rollout_meta: the dictionary containing metadata for each rollout, including the path to the video file.
    :param int max_per_dim: the maximum number of highlights to be recorded for each dimension for above and below outliers.
    :param float iqr_mul: the IQR multiplier to determine outliers.
    :param record_timesteps: the number of timesteps to be recorded in each highlight video.
    :param fade_ratio: the ratio of frames to which apply a fade-in/out effect.
    :param int processes: the number of parallel processes used to process highlights.
    :param img_format: the image format of the rollout plots to be produced.
    :param str videos_dir: path to the location of videos.
    """
    videos_dir = videos_dir or ''  # make sure we have a video path even if empty

    dfs = []
    for dim in dimensions:
        logging.info('_____________________________________')
        logging.info(f'Processing highlights for {dim}...')

        _output_dir = os.path.join(output_dir, dim.lower().replace(' ', '-'))
        create_clear_dir(_output_dir)

        # compute outliers from interestingness dataframe
        lower_idxs, upper_idxs = _find_outliers(df, dim, iqr_mul)

        videos_files: Dict[str, str] = {
            rollout['id']: rollout['video_file']
            if os.path.isfile(rollout['video_file'])
            else os.path.join(videos_dir, rollout['video_file'])
            for rollout in rollout_meta['rollouts']}

        # saves highlight information
        for idxs, label in [(lower_idxs, 'low'), (upper_idxs, 'high')]:
            if len(idxs) == 0:
                logging.info(f'No {label} {dim} highlights found')
                continue

            logging.info(f'Getting info for {len(idxs)} {label} {dim} highlights...')
            df_ = _get_highlights_info(df, idxs, dim, label)

            logging.info(f'Saving {label} {dim} video highlights...')
            _save_highlights(df_, dim, label, max_per_dim, _output_dir, figures, videos_files,
                             record_timesteps, fade_ratio, processes, img_format)
            dfs.append(df_)

    logging.info('Putting all highlights together...')
    df = pd.concat(dfs, axis=0)
    file_path = os.path.join(output_dir, 'highlights.csv')
    df.to_csv(file_path, index=False)
    logging.info(f'Saved all highlights information (total {len(df)}) to {file_path}')


def _get_highlights_info(df: pd.DataFrame, idxs: List[int], dim: str, label: str):
    # creates dataframe with info for all highlights
    _df = df.loc[idxs, [ROLLOUT_ID_COL, TIMESTEP_COL, dim]].copy()
    _df.rename(columns={dim: VALUE_COL}, inplace=True)
    _df[DIMENSION_COL] = dim
    _df[LABEL_COL] = label
    _df[ROLLOUT_LEN_COL] = 0
    for idx, rollout_info in tqdm.tqdm(_df.iterrows(), total=len(_df)):
        _df.loc[idx, ROLLOUT_LEN_COL] = len(df[df[ROLLOUT_ID_COL] == rollout_info[ROLLOUT_ID_COL]])
    return _df[[DIMENSION_COL, VALUE_COL, LABEL_COL, ROLLOUT_ID_COL, TIMESTEP_COL, ROLLOUT_LEN_COL]]


def _find_outliers(df: pd.DataFrame, dim: str, iqr_mul: float) -> Tuple[List[int], List[int]]:
    """
    Based on: https://stackoverflow.com/a/69248173/16031961
    """
    # get outliers based on IQR
    q1 = df[dim].quantile(.25)
    q3 = df[dim].quantile(.75)
    iqr = q3 - q1
    ll = q1 - (iqr_mul * iqr)
    ul = q3 + (iqr_mul * iqr)
    upper_outliers = df[df[dim] > ul]
    lower_outliers = df[df[dim] < ll]

    # select min and max from each rollout
    upper_idxs = [_df[dim].idxmax() for _, _df in upper_outliers.groupby(ROLLOUT_ID_COL)]
    lower_idxs = [_df[dim].idxmin() for _, _df in lower_outliers.groupby(ROLLOUT_ID_COL)]

    # sort
    upper_idxs.sort(key=lambda idx: df.loc[idx, dim], reverse=True)
    lower_idxs.sort(key=lambda idx: df.loc[idx, dim])

    # simply select the first from each list
    return lower_idxs, upper_idxs


def _save_highlights(df: pd.DataFrame, dim: str, label: str, max_outliers: int,
                     output_dir: str, figures: Dict[str, Dict[str, str]], video_dirs: Dict[str, str],
                     record_timesteps: int, fade_ratio: float, processes: int, img_format: str):
    df.to_csv(os.path.join(output_dir, get_clean_filename(f'{dim}-{label}.csv')), index=False)

    logging.info(f'Generating videos and saving rollout plots for {label} highlights...')
    args = []
    for _, rollout_info in tqdm.tqdm(df.iloc[: min(len(df), max_outliers)].iterrows(), total=max_outliers):
        rollout_id = rollout_info[ROLLOUT_ID_COL]
        t = rollout_info[TIMESTEP_COL]
        length = rollout_info[ROLLOUT_LEN_COL]
        fig = from_json(figures[rollout_id][dim]) if rollout_id in figures and dim in figures[rollout_id] else None
        video_file = video_dirs[rollout_id] if rollout_id in video_dirs else None
        args.append((rollout_id, t, length, dim, label, output_dir, fig, video_file,
                     record_timesteps, fade_ratio, img_format))

    run_parallel(_save_highlight, args, processes, use_tqdm=True)


def _save_highlight(rollout_id: str, t: int, length: int, dim: str, label: str,
                    output_dir: str, fig: Optional[go.Figure], video_file: Optional[str],
                    record_timesteps: int, fade_ratio: float, img_format: str):
    dummy_plotly()

    # saves figures corresponding to each highlight, annotated
    if fig is not None:
        x = t
        y = fig.data[0].y[x]
        fig.add_annotation(x=x, y=y, showarrow=False, xanchor='right', yanchor='bottom', xshift=-5, yshift=5,
                           font=dict(color='black', size=13),
                           text=f'{label.title()} {dim.title()}')
        fig.add_scatter(x=[x], y=[y], mode='markers', name=dim.title(),
                        marker=dict(size=20, color=fig.data[0].line.color, symbol='star'), )
        format_and_save_plot(
            fig, title=fig.layout.title.text,
            output_img=os.path.join(output_dir, get_clean_filename(f'{dim}-{label}-{rollout_id}.{img_format}')),
            show_legend=False)

    # select highlight from video file
    if video_file is None or not os.path.isfile(video_file):
        logging.info(f'Could not find video file of rollout {rollout_id}: {video_file}')
        return

    metadata = skvideo.io.ffprobe(video_file)
    n_frames = int(metadata['video']['@nb_frames'])
    if n_frames - 1 != length:
        logging.warning(f'Skipping highlight for {rollout_id}: '
                        f'length of video ({n_frames - 1}) does not match length of rollout ({length})')
        return

    video_frames = skvideo.io.vread(video_file)  # load all video frames

    t += 1  # skip first frame
    half_len = int((record_timesteps - 1) / 2)
    start_t = max(0, t - half_len)
    end_t = min(t + half_len, len(video_frames) - 1)
    buffer = [Image.fromarray(f) for f in video_frames[start_t:end_t + 1]]
    buffer = fade_video(buffer, fade_ratio)
    if len(buffer) == 0:
        logging.warning(f'Skipping highlight for {rollout_id}: insufficient frames')
        return

    # save video clip and still frame
    fps = int(eval(metadata['video']['@r_frame_rate']))
    save_video(buffer, os.path.join(output_dir, get_clean_filename(f'{dim}-{label}-{rollout_id}.mp4')), fps)
    buffer[half_len].save(os.path.join(output_dir, get_clean_filename(f'{dim}-{label}-{rollout_id}.png')))
