from typing import Dict, Set, List

import numpy as np
import pandas as pd

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def get_equal_columns(df: pd.DataFrame) -> List[Set[str]]:
    """
    Checks for columns in the given dataset that are equal, i.e., contain the same values for each row.
    Then computes the equality groups, i.e., the unique groups of columns that are equivalent with one another.
    :param pd.DataFrame df: the dataframe for which to check the equal columns.
    :rtype: list[set[str]]
    :return: a list containing sets of names of columns whose values are equal to one another.
    """
    eq_cols: Dict[str, Set[str]] = {}
    for i in range(len(df.columns)):
        i_vals = df.iloc[:, i]
        i_col = df.columns[i]
        for j in range(i + 1, len(df.columns)):
            j_vals = df.iloc[:, j]
            j_col = df.columns[j]
            if i_col in eq_cols and j_col in eq_cols[i_col]:
                continue  # already tested
            if np.array_equal(i_vals, j_vals):
                # updates equality sets
                if i_col not in eq_cols:
                    eq_cols[i_col] = {i_col}
                if j_col not in eq_cols:
                    eq_cols[j_col] = {j_col}
                eq_cols[i_col].add(j_col)
                for z_col in eq_cols[i_col]:
                    eq_cols[z_col].update(eq_cols[i_col])

    # get unique equality groups
    eq_groups = []
    for eq_group in eq_cols.values():
        if eq_group not in eq_groups:
            eq_groups.append(eq_group)
    return eq_groups
