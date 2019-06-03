"""Utilities for stacking/unstacking dimensions
"""
from collections import OrderedDict
from typing import Hashable, MutableMapping, TypeVar
import pandas
import xarray


T = TypeVar('T', xarray.DataArray, xarray.Dataset)


def proper_unstack(array: T, dim: Hashable) -> T:
    """Work around an issue in xarray that causes the data to be sorted
    alphabetically by label on unstack():

    `<https://github.com/pydata/xarray/issues/906>`_

    Also work around issue that causes string labels to be converted to
    objects:

    `<https://github.com/pydata/xarray/issues/907>`_

    :param array:
        xarray.DataArray or xarray.Dataset to unstack
    :param str dim:
        Name of existing dimension to unstack
    :returns:
        xarray.DataArray or xarray.Dataset with unstacked dimension
    """
    # Regenerate Pandas multi-index to be ordered by first appearance
    mindex = array.coords[dim].to_pandas().index

    levels = []
    codes = []
    try:
        mindex_codes = mindex.codes
    except AttributeError:
        # pandas < 0.24
        mindex_codes = mindex.labels

    for levels_i, codes_i in zip(mindex.levels, mindex_codes):
        level_map = OrderedDict()  # type: MutableMapping[Hashable, int]

        for code in codes_i:
            if code not in level_map:
                level_map[code] = len(level_map)

        levels.append([levels_i[k] for k in level_map.keys()])
        codes.append([level_map[k] for k in codes_i])

    mindex = pandas.MultiIndex(levels, codes, names=mindex.names)
    array = array.copy()
    array.coords[dim] = mindex

    # Invoke builtin unstack
    array = array.unstack(dim)

    # Convert numpy arrays of Python objects to numpy arrays of C floats, ints,
    # strings, etc.
    for dim in mindex.names:
        if array.coords[dim].dtype == object:
            array.coords[dim] = array.coords[dim].values.tolist()

    return array
