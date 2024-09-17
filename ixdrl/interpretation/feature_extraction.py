import itertools as it
from enum import IntEnum
from typing import Optional, List, Dict

import gymnasium as gym
import numpy as np
import pandas as pd

from ixdrl import Rollout
from ixdrl.analysis import ROLLOUT_ID_COL, TIMESTEP_COL
from ixdrl.util.gym import RepeatedDiscrete, is_spatial

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

SPATIAL_DIMENSIONS = 2  # min number of spatial dimensions for feature layers (2D data)
CONNECTIVITY = 8  # 8 or 4, connectivity for extracting components/objects of spatial layers


def extract_features_from_rollout(rollout: Rollout) -> pd.DataFrame:
    """
    Automatically extract a set of descriptive observation features from the given rollout (interaction data).
    :param Rollout rollout: the rollout containing the observations for which to extract the features.
    :rtype: pd.DataFrame
    :return: a pandas dataframe containing the extracted observation features (columns) for each timestep instance
    (row). Columns `Rollout` and `Timestep` are added to the dataframe, containing the rollout ID and the timestep
    number of each instance, respectively.
    """
    rollout_df = extract_features(rollout.data.observation, rollout.observation_space,
                                  name=None, labels=rollout.observation_labels)

    # add rollout id, timestep columns to later align features with interestingness
    rollout_df[ROLLOUT_ID_COL] = rollout.rollout_id
    rollout_df[TIMESTEP_COL] = rollout.data.timesteps
    rollout_df = rollout_df[[ROLLOUT_ID_COL, TIMESTEP_COL] + list(rollout_df.columns[:-2])]
    return rollout_df


def extract_features(data: np.ndarray, space: gym.Space,
                     name: Optional[str] = None, labels: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extracts descriptive features from the given data according to the provided space definition.
    :param np.ndarray data: the data from which to extract the features, shaped: (batch, dim1, dim2, ...).
    :param gym.Space space: the gym space definition that helps interpret and extract features from the given data.
    :param str name: the name of the space for which we are extracting features (used for feature naming).
    :param str labels: the names of the different dimensions that this space offers for feature extraction (used for feature naming).
    :rtype: pd.DaraFrame
    :return: a pandas dataframe containing the extracted observation features (columns) for each timestep instance (row).
    """
    # data is in batch format, shape: (batch, dim1, dim2, ...)

    # extracts features depending on the data type
    if isinstance(space, gym.spaces.Discrete) and len(data.shape) == 2 and data.shape[1] == space.n:
        # if discrete, simply one-hot encode data
        df = pd.get_dummies(data.astype(str)).sum(axis=0, skipna=True)

        # try to rename columns
        if labels is None or len(labels) != space.n:
            dim_name = name if name is not None else labels[0] if labels is not None else 'Cat'
            labels = [f'{dim_name}={i}' for i in range(space.n)]  # assumes category labels are the indices
        rename_dict = {}
        labels = list(labels)  # make copy
        for i, label in enumerate(labels):
            if isinstance(label, IntEnum):
                rename_dict[str(label.value)] = label.name  # replace int value with label
                labels[i] = label.name  # set only enum name in label
            else:
                rename_dict[str(i)] = str(label)  # try to replace index value with label

        return df.rename(columns=rename_dict)[labels].copy()  # rename and select only columns from given labels

    if isinstance(space, gym.spaces.Tuple) and len(data.shape) > 1 and data.shape[1] == len(space.spaces):
        # if tuple, concat the features for the different subspaces
        if labels is None:
            labels = [f'Subspace {i}' for i in range(len(space.spaces))]  # generates dummy labels if not provided
        dfs = [extract_features(data[:, i], space.spaces[i], name=labels[i], labels=None)
               for i in range(len(space.spaces))]
        return pd.concat(dfs, axis=1)  # concatenate resulting features columns

    elif isinstance(space, RepeatedDiscrete):
        cat_labels = space.labels if space.labels is not None else np.arange(np.iinfo(space.dtype).max)
        spatial, idxs = is_spatial(space, SPATIAL_DIMENSIONS)

        if spatial:
            # if discrete spatial, extract spatial features for each spatial "layer"
            axes = list(np.array(idxs) + 1)
            axes = axes[:-SPATIAL_DIMENSIONS] + [0] + axes[-SPATIAL_DIMENSIONS:]
            data = data.transpose(axes)  # data is batch mode, so swap such that shape: (dims*, batch, w, h)

            dfs: List[pd.DataFrame] = []
            if len(idxs) == 2:
                dfs.append(extract_categorical_spatial_features(data, name, cat_labels))  # only one layer
            else:
                # extract features for each layer
                for comb in it.product(*[np.arange(s) for s in np.array(space.shape)[idxs][:-SPATIAL_DIMENSIONS]]):
                    dfs.append(extract_categorical_spatial_features(data[comb], f'{name}{comb}', cat_labels))

            return pd.concat(dfs, axis=1)  # concatenate features for different layers

        else:
            # otherwise simply one-hot encode data for each dimension
            if labels is None:
                labels = [f'Dim {i}' for i in range(space.shape[0])]  # generates dummy labels if not provided
            dfs = [
                extract_features(data[i], gym.spaces.Discrete(len(cat_labels)), name=labels[i], labels=cat_labels)
                for i in range(len(labels))
            ]
            return pd.concat(dfs, axis=1)  # concatenate features for different dimensions

    elif isinstance(space, gym.spaces.Box):
        spatial, idxs = is_spatial(space)

        if spatial:
            # if spatial, extract continuous spatial features
            axes = list(np.array(idxs) + 1)
            axes = axes[:-SPATIAL_DIMENSIONS] + [0] + axes[-SPATIAL_DIMENSIONS:]
            data = data.transpose(axes)  # data is batch mode, so swap such that shape: (dims*, batch, w, h)

            dfs: List[pd.DataFrame] = []
            if len(idxs) == 2:
                dfs.append(extract_continuous_spatial_features(data, name))
            else:
                layer_names = [labels[idx] if labels is not None else f'Dim{i}'
                               for i, idx in enumerate(idxs[:-2])]
                for comb in it.product(*[np.arange(s) for s in np.array(space.shape)[idxs][:-2]]):
                    layer_name = name if name is not None else ''
                    layer_name += ''.join(f'{layer_names[i]} {comb[i]}' for i in range(len(comb)))
                    dfs.append(extract_continuous_spatial_features(data[comb], layer_name))

            return pd.concat(dfs, axis=1)  # concatenate results for different layers
        else:
            # otherwise simply return feature values
            if labels is None:
                labels = [f'Dim {i}' for i in range(space.shape[0])]  # generates dummy labels if not provided
            return pd.DataFrame(data, columns=labels)  # data already in shape: (batch, n-dims)

    else:
        # some mismatch or unsupported data type
        raise ValueError(f'Could not process space of type: {space}, '
                         f'or data (shape: {data.shape}) incompatible with space.')


def extract_continuous_spatial_features(data: np.ndarray, name: str) -> pd.DataFrame:
    """
    Extracts descriptive features from the given data, assumed to be spatially structured and continuous valued.
    In particular, identifies the different groups/regions in each provided layer using OpenCV.
    :param np.array data: the spatial data from which to extract the features, shaped: (batch, width, height).
    :param str name: the name of the data layer, used for feature naming.
    :rtype: pd.DataFrame
    :return: a pandas dataframe containing the extracted observation features (columns) for each timestep instance (row).
    """
    # data shape is (batch, W, H)
    assert len(data.shape) == 3, f'Input data should have shape: (batch, W, H), but {data.shape} given.'

    # extract components/regions features for each "grayscale" frame in batch
    features: List[Dict[str, float]] = []
    for i in range(data.shape[0]):
        features.append(extract_image_component_features(data[i], name))
    return pd.DataFrame(features)  # concatenate features (columns) for all frames (rows)


def extract_categorical_spatial_features(data: np.ndarray, name: str, labels: List[str]) -> pd.DataFrame:
    """
    Extracts descriptive features from the given data, assumed to be spatially structured and categorical.
    In particular, identifies the different groups/regions in each provided layer for each category using OpenCV.
    :param np.array data: the spatial data from which to extract the features, shaped: (batch, width, height).
    :param str name: the name of the data layer, used for feature naming.
    :param list[str] labels: the names of each category.
    :rtype: pd.DataFrame
    :return: a pandas dataframe containing the extracted observation features (columns) for each timestep instance (row).
    :return:
    """
    # data shape is (batch, W, H)
    assert len(data.shape) == 3, f'Input data should have shape: (batch, W, H), but {data.shape} given.'

    # compute discrete values to labels mapping
    value_label_map = {}
    for i, label in enumerate(labels):
        if isinstance(label, IntEnum):
            value_label_map[label.value] = label.name  # replace int value with label
        elif 'int' in str(type(label)) or 'float' in str(type(label)):
            value_label_map[label] = str(label)  # replace int value with string
        elif isinstance(label, str) and label.replace('.', '', 1).isdigit():
            value_label_map[float(label)] = label  # replace numeric value with string
        else:
            value_label_map[i] = str(label)  # try to replace index value with label

    features: List[pd.DataFrame] = []
    for value, label in value_label_map.items():
        # gets spatial features for each category value
        _data = (data == value).astype(np.uint8)  # convert to binary "image"
        _name = f'{name}={label}'
        if not np.any(_data):
            continue  # category was not found in data, ignore

        # extract components/regions features for each binary frame in batch
        cat_features: List[Dict[str, float]] = []
        for i in range(_data.shape[0]):
            cat_features.append(extract_image_component_features(_data[i], _name))

        features.append(pd.DataFrame(cat_features))  # concatenate features (columns) for all frames (rows)

    return pd.concat(features, axis=1)  # concatenate all features for each category


def extract_image_component_features(img: np.array, prefix: str, ignore_background: bool = True) -> Dict[str, float]:
    """
    Gets features from the mean and standard deviation of various image properties extracted from the different
    components (separate objects or regions) detected in the given image. Currently, the following component
    properties are computed: width, height, area, centroid X coord, centroid Y coord.
    :param np.array img: the original image containing the objects/regions from which to extract the properties. Will
    be converted to a numpy array of `uint8` type, so should be a grayscale or binary image.
    :param str prefix: the prefix to be added to the extracted features.
    :param bool ignore_background: whether to ignore the background when computing the stats. This corresponds to
    the largest detected object/region in the image and whose value is equal to `np.min(img)`.
    :rtype: dict[str, float]
    :return: a dictionary containing the extracted properties' features (values) for each feature label (key).
    """
    import cv2  # lazy loading

    # detects separate components / objects / regions
    img = np.asarray(img)
    if img.dtype == bool or (img.dtype == float and np.max(img) <= 1):
        img = np.asarray(img * 255, dtype=np.uint8)  # assume it's 0-1 normalized, so un-normalize
    else:
        img = img.astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=CONNECTIVITY)

    # check background
    start_idx = 1 if ignore_background and np.any(img[np.where(labels == 0)] == np.min(img)) else 0

    # get component moments' features (possibly ignore 1st component which is background)
    stats = {
        'Width': stats[start_idx:, cv2.CC_STAT_WIDTH].astype(np.float64),
        'Height': stats[start_idx:, cv2.CC_STAT_HEIGHT].astype(np.float64),
        'Area': stats[start_idx:, cv2.CC_STAT_AREA].astype(np.float64),
        'Centroid X': centroids[start_idx:, 0].astype(np.float64),
        'Centroid Y': centroids[start_idx:, 1].astype(np.float64)
    }

    # get means of stats over components/regions
    features = {f'{prefix} Groups': n - start_idx}
    for lbl, stat in stats.items():
        features[f'{prefix} {lbl} Mean'] = np.mean(stat) if len(stat) > 0 else np.nan
        features[f'{prefix} {lbl} Std'] = np.std(stat) if len(stat) > 0 else np.nan
    return features
