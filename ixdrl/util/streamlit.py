import logging
import numbers
import time
import threading
import numpy as np
import streamlit as st
from typing import Callable, Iterable, Optional, Dict, Tuple, Any
from streamlit.elements.media import marshall_video
from streamlit.proto.Video_pb2 import Video as VideoProto
from streamlit_player import st_player
from streamlit_autorefresh import st_autorefresh
from .video import get_video_metadata, VideoMetadata

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

# session params
FRAME_SESS_PARAM = 'cur-frame'
FRAME_RANGE_SESS_PARAM = 'frame-range'
PLAYING_SESS_PARAM = 'is-playing'
VIDEO_URL_SESS_PARAM = 'video-url'


def format_markdown(obj: Any, float_dec: int = 2) -> str:
    """
    Converts the given object into a markdown string.
    :param obj: the object to be converted.
    :param int float_dec: number of decimal places for float types.
    :rtype: str
    :return: a string in the markdown format representing the given object.
    """
    cur_str = ''
    if isinstance(obj, str):
        return obj  # already a string
    elif isinstance(obj, int):
        return str(obj)
    elif isinstance(obj, (float, np.float32)):
        return f'{obj:.{float_dec}f}'
    elif isinstance(obj, dict):
        # creates an unordered list
        for k, v in obj.items():
            cur_str += f'- **{format_markdown(k, float_dec)}:** {format_markdown(v, float_dec)}\n'
    elif isinstance(obj, list):
        # creates an ordered list
        for i, v in enumerate(obj):
            cur_str += f'{i}. {format_markdown(v, float_dec)}\n'
    elif isinstance(obj, tuple) and len(obj) == 2 and all(isinstance(v, numbers.Number) for v in obj):
        # assume it's a mean and std deviation
        cur_str += f'{format_markdown(obj[0], float_dec)} ± {format_markdown(obj[1], float_dec)}'
    else:
        cur_str = str(obj)

    return cur_str


def execute_async(func: Callable, args: Iterable) -> threading.Thread:
    """
    Executes the given function asynchronously (in a new thread) and registers the thread in streamlit such that the
    function can interact with the page.
    See: https://discuss.streamlit.io/t/how-to-run-a-subprocess-programs-using-thread-inside-streamlit/2440/2
    :param func: the function to be executed.
    :param args: the arguments to pass along to the function.
    :rtype: threading.Thread
    :return: the thread used to call the function asynchronously. The main streamlit thread should store this and join
    the thread after all UI elements are added.
    """
    t = threading.Thread(target=func, args=args)
    st.report_thread.add_report_ctx(t)
    t.start()
    return t


def temp_widget(name: str, timeout: int, *args) -> threading.Thread:
    """
    Asynchronously adds a widget inside a temporary `streamlit.empty` widget.
    :param str name: the name of the widget function to be added, e.g., `"text"`, `"info"`, etc.
    :param int timeout: the number of seconds after which the `empty` widget is removed/closed.
    :param args: the arguments to be passed along to the widget.
    :rtype: threading.Thread
    :return: the thread used to create the widget asynchronously. The main streamlit thread should store this and join
    the thread after all UI elements are added.
    """
    return execute_async(_temp_widget, (name, timeout, *args))


def _temp_widget(name: str, timeout: int, *args):
    output = st.empty()  # the removable, empty widget
    if hasattr(output, name) and isinstance(getattr(output, name), Callable):
        getattr(output, name)(*args)  # create sub-widget
    time.sleep(timeout)  # wait for timeout
    output.empty()  # close/remove widget


def _load_video(url: str, fmt: str = "video/mp4") -> str:
    """
    Uses streamlit's functions to load a video and gets the resulting local url.
    :param str url: the url/path to the video to be loaded.
    :param str fmt: the video's MIME type format.
    :rtype: str
    :return: the url of the serialized local file.
    """
    video_proto = VideoProto()
    coordinates = '[]'
    marshall_video(coordinates, video_proto, url, fmt)
    return video_proto.url


def _get_video_url(player_id: str) -> str:
    """
    Loads the current video URL from the session.
    :param str player_id: the video player's identifier.
    :rtype: float
    :return: the video URL.
    """
    return st.session_state[f'{player_id}-{VIDEO_URL_SESS_PARAM}'] \
        if f'{player_id}-{VIDEO_URL_SESS_PARAM}' in st.session_state else None


def _set_video_url(player_id: str, url: str):
    """
    Stores the given video URL in the session.
    :param str player_id: the video player's identifier.
    :param str url: the video URL.
    :return:
    """
    st.session_state[f'{player_id}-{VIDEO_URL_SESS_PARAM}'] = url


def _get_cur_frame(player_id: str) -> float:
    """
    Loads the current frame from the session.
    :param str player_id: the video player's identifier.
    :rtype: float
    :return: the current frame.
    """
    return st.session_state[f'{player_id}-{FRAME_SESS_PARAM}']


def _set_cur_frame(player_id: str, frame: float):
    """
    Stores the given frame in the session.
    :param str player_id: the video player's identifier.
    :param float frame: the current frame.
    :return:
    """
    st.session_state[f'{player_id}-{FRAME_SESS_PARAM}'] = frame


def _get_frame_range(player_id: str) -> Tuple[float, float]:
    """
    Loads the frame range from the session.
    :param str player_id: the video player's identifier.
    :rtype: (float, float)
    :return: the frame range.
    """
    return st.session_state[f'{player_id}-{FRAME_RANGE_SESS_PARAM}']


def _set_frame_range(player_id: str, frame_range: Tuple[float, float]):
    """
    Stores the given frame range in the session.
    :param str player_id: the video player's identifier.
    :param (float, float) frame_range: the frame range.
    :return:
    """
    _set_cur_frame(player_id, frame_range[0])
    st.session_state[f'{player_id}-{FRAME_RANGE_SESS_PARAM}'] = frame_range


def _is_playing(player_id: str) -> bool:
    """
    Loads the variable from the session informing about whether the player should be playing.
    :param str player_id: the video player's identifier.
    :rtype: bool
    :return: whether the player should be playing.
    """
    return st.session_state[f'{player_id}-{PLAYING_SESS_PARAM}']


def _set_playing(player_id: str, playing: bool):
    """
    Stores in the session a variable informing about whether the player should be playing.
    :param str player_id: the video player's identifier.
    :param bool playing: whether the player should be playing.
    :return:
    """
    st.session_state[f'{player_id}-{PLAYING_SESS_PARAM}'] = playing


def reset_video_player(player_id: str):
    """
    Deletes all session cookies associated with the player.
    :param str player_id: the video player's identifier.
    :return:
    """
    if f'{player_id}-{PLAYING_SESS_PARAM}' in st.session_state:
        del st.session_state[f'{player_id}-{PLAYING_SESS_PARAM}']
    if f'{player_id}-{FRAME_SESS_PARAM}' in st.session_state:
        del st.session_state[f'{player_id}-{FRAME_SESS_PARAM}']
    if f'{player_id}-{FRAME_RANGE_SESS_PARAM}' in st.session_state:
        del st.session_state[f'{player_id}-{FRAME_RANGE_SESS_PARAM}']
    if f'{player_id}-{VIDEO_URL_SESS_PARAM}' in st.session_state:
        del st.session_state[f'{player_id}-{VIDEO_URL_SESS_PARAM}']


def _create_frame_range_slider(frame_range_container: st.delta_generator.DeltaGenerator,
                               player_id: str,
                               metadata: VideoMetadata,
                               min_frame: int = 0,
                               max_frame: int = -1,
                               label: str = 'Frame range:') -> Tuple[int, int]:
    """
    Adds a streamlit range slider widget in the given container, where the range is retrieved from the session.
    :param st.delta_generator.DeltaGenerator frame_range_container: the container in which to create the slider.
    :param str player_id: the video player's identifier.
    :param dict[str] metadata: the current video's metadata.
    :param int min_frame: the index of the minimum frame to be played by the video player.
    :param int max_frame: the index of the maximum frame to be played by the video player. `-1` means last frame.
    :param str label: the label to appear above the slider.
    :rtype: (int, int)
    :return: the selected range from the slider widget.
    """
    val = _get_frame_range(player_id)
    val = (int(val[0]), int(val[1]))
    min_frame = max(0, min(metadata.total_frames - 2, min_frame))
    max_frame = metadata.total_frames - 1 if max_frame < 1 else min(metadata.total_frames - 1, max_frame)
    return frame_range_container.slider(label, min_frame, max_frame, val)


def _create_cur_frame_slider(cur_frame_container: st.delta_generator.DeltaGenerator,
                             player_id: str,
                             label: str = 'Current frame:') -> int:
    """
    Adds a streamlit value slider widget in the given container, where the value is the current frame retrieved from
    the session.
    :param st.delta_generator.DeltaGenerator cur_frame_container: the container in which to create the slider.
    :param str player_id: the video player's identifier.
    :param str label: the label to appear above the slider.
    :rtype: int
    :return: the selected value from the slider widget.
    """
    min_frame, max_frame = _get_frame_range(player_id)
    val = _get_cur_frame(player_id)
    return cur_frame_container.slider(label, int(min_frame), int(max_frame), int(val))


def _create_player(video_container: st.delta_generator.DeltaGenerator,
                   player_id: str,
                   metadata: VideoMetadata,
                   height: Optional[int] = None,
                   playback_rate: float = 1.) -> Dict:
    """
    Adds a streamlit video player widget in the given container, where the play time is set according to the range
    retrieved from the session.
    See: https://github.com/okld/streamlit-player
    :param st.delta_generator.DeltaGenerator video_container: the container in which to create the video player.
    :param str player_id: the video player's identifier.
    :param dict[str] metadata: the current video's metadata.
    :param int height: the player's height.
    :param float playback_rate: the video playback rate/speed.
    :rtype: dict[str]
    :return: a dictionary containing the player's widget events.
    """
    with video_container:
        # converts range to min and max time values and updates player
        min_frame, max_frame = _get_frame_range(player_id)
        min_frame = max(min_frame, _get_cur_frame(player_id))  # selects higher of min range and cur frame
        min_t = (min_frame * metadata.total_secs) / (metadata.total_frames - 1)
        max_t = (max_frame * metadata.total_secs) / (metadata.total_frames - 1)
        url = _get_video_url(player_id)
        url = f'{_load_video(url)}#t={min_t},{max_t}'
        logging.debug(f'Created new player with URL: {url}')
        event = st_player(
            url, playing=_is_playing(player_id), playback_rate=playback_rate, height=height,
            controls=False, loop=False,
            config=dict(file=dict(attributes=dict(style={'background-color': '#0e1117',  # default back color
                                                         'width': '100%',
                                                         'height': '100%'}))),
        )
        return event


def _on_play_button(player_id):
    _set_playing(player_id, not _is_playing(player_id))


def create_video_player(url: str,
                        video_container: Optional[st.delta_generator.DeltaGenerator] = None,
                        cur_frame_container: Optional[st.delta_generator.DeltaGenerator] = None,
                        frame_range_container: Optional[st.delta_generator.DeltaGenerator] = None,
                        play_button_container: Optional[st.delta_generator.DeltaGenerator] = None,
                        reset_button_container: Optional[st.delta_generator.DeltaGenerator] = None,
                        height: Optional[float] = 1.,
                        playback_rate: Optional[float] = 1.,
                        player_id: str = 'video-player',
                        start_playing: bool = False,
                        min_frame: int = 0,
                        max_frame: int = -1,
                        frame_range_label: str = 'Frame range:',
                        cur_frame_label: str = 'Current frame:') -> Tuple[Tuple[int, int], int]:
    """
    Creates a set of streamlit widgets for visualizing and controlling the playback of video files.
    :param str url: the url/path to the video file to be played in the player.
    :param st.delta_generator.DeltaGenerator video_container: the container in which to create the video player.
    :param st.delta_generator.DeltaGenerator cur_frame_container: the container in which to create the slider.
    :param st.delta_generator.DeltaGenerator frame_range_container: the container in which to create the slider.
    :param st.delta_generator.DeltaGenerator play_button_container: the container in which to create the play button.
    :param st.delta_generator.DeltaGenerator reset_button_container: the container in which to create the reset button.
    :param int height: the video's height ratio.
    :param float playback_rate: the video's playback rate/speed.
    :param str player_id: the video player's identifier.
    :param bool start_playing: whether the player should start playing when loaded.
    :param int min_frame: the index of the minimum frame to be played by the video player.
    :param int max_frame: the index of the maximum frame to be played by the video player. `-1` means last frame.
    :param str frame_range_label: the label to appear above the frame range slider.
    :param str cur_frame_label: the label to appear above the current frame slider.
    :rtype: ((int, int), int)
    :return: a tuple containing:
        - the selected frame range
        - the select frame
    """
    # checks session and sets defaults
    if _get_video_url(player_id) is None or _get_video_url(player_id) != url:
        reset_video_player(player_id)  # different video, reset session
        _set_playing(player_id, start_playing)
        _set_frame_range(player_id, (min_frame, max_frame))
        _set_cur_frame(player_id, min_frame)
        _set_video_url(player_id, url)

        # creates the player's layout containers if not given
    video_container = video_container or st.container()
    frame_range_container = frame_range_container or st.empty()
    cur_frame_container = cur_frame_container or st.empty()
    play_button_container = play_button_container or st.empty()
    reset_button_container = reset_button_container or st.empty()

    # clears all containers just in case
    video_container.empty()
    frame_range_container.empty()
    cur_frame_container.empty()
    play_button_container.empty()

    # gets video info
    metadata = get_video_metadata(url)
    height = int(metadata.height * height)

    # add play button
    if _is_playing(player_id):
        play_button_container.button('⏯', on_click=_on_play_button, args=(player_id,), key='play_button_pause')
    else:
        play_button_container.button('▶️️', on_click=_on_play_button, args=(player_id,), key='play_button_play')

    # add reset button
    if reset_button_container.button('🔄', key='reset_button'):
        reset_video_player(player_id)

    # add sliders
    prev_range = _get_frame_range(player_id)
    prev_frame = _get_cur_frame(player_id)
    logging.debug(f'Creating slider: {_get_frame_range(player_id)}')
    cur_range = _create_frame_range_slider(
        frame_range_container, player_id, metadata, min_frame, max_frame, frame_range_label)
    cur_frame = _create_cur_frame_slider(cur_frame_container, player_id, cur_frame_label)
    if not _is_playing(player_id):
        # if player's not playing, then update sliders
        _set_frame_range(player_id, cur_range)
        if prev_range != cur_range:
            logging.debug(f'Changed range to: {cur_range}')
            _create_frame_range_slider(
                frame_range_container, player_id, metadata, min_frame, max_frame, frame_range_label)
            cur_frame = cur_range[0]

        _set_cur_frame(player_id, cur_frame)
        if prev_frame != cur_frame:
            logging.debug(f'Changed frame to: {cur_frame}')
            _create_cur_frame_slider(cur_frame_container, player_id, cur_frame_label)

    # add player
    _create_player(video_container, player_id, metadata, height, playback_rate)

    # update frame slider until reaching end frame
    while _is_playing(player_id):
        logging.debug(f'Playing at: {_get_cur_frame(player_id)}')

        # sleep 1 sec, assume frames passed according to fps and playback rate
        time.sleep(1)
        min_frame, max_frame = _get_frame_range(player_id)
        delta = metadata.fps * playback_rate
        if _get_cur_frame(player_id) + delta > max_frame:
            # if maximum reached, stop playing and refresh
            _set_playing(player_id, False)
            _set_cur_frame(player_id, min_frame)
            logging.debug('Video stopped')
            st_autorefresh(interval=0, limit=2)
        else:
            # otherwise increment frame slider
            _set_cur_frame(player_id, _get_cur_frame(player_id) + delta)
            _create_cur_frame_slider(cur_frame_container, player_id, cur_frame_label)

    return cur_range, cur_frame
