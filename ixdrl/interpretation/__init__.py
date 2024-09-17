import re
import pandas as pd
import plotly.express as px
from typing import Tuple

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

# discrete color palette for interestingness dimensions, consistent across feature importance plots
DIMS_PALETTE = px.colors.qualitative.Light24  # Dark24
FEATURES_LABEL_FONT_SIZE = 12  # font size for feature labels for importance plots


def get_clean_filename(name: str) -> str:
    """
    Utility method that return a properly formatted name for a file, resulting in a lower case, non-spaced name.
    :param str name: the original file name to be converted.
    :rtype: str
    :return: the cleaned/sanitized file name.
    """
    return re.sub(r'[ =_]', '-', name.lower())


def remove_nan_targets(x: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Utility method that removes rows where the given target has a NaN value.
    :param pd.DataFrame x: the features dataset.
    :param pd.DataFrame y: the targets dataset, assumed to be single-column and index-aligned with `x`.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    :return: a tuple containing the original datasets without the rows in which the targets have a Nan value.
    """
    nan_idxs = y.loc[y.iloc[:, 0].isna(), :].index
    return x.drop(nan_idxs), y.drop(nan_idxs)  # assumes x is index-aligned with y
