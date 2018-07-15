import xarray
import pandas as pd
from dask.base import tokenize
from dask.delayed import Delayed
from dask import sharedict
from .kernels import csv as kernels


__all__ = ('to_csv', )


def to_csv(x, path_or_buf, **kwargs):
    if not isinstance(x, xarray.DataArray):
        raise ValueError("first argument must be a DataArray")

    if not x.chunks:
        x.to_pandas().to_csv(path_or_buf, **kwargs)
        return

    if not isinstance(path_or_buf, str):
        raise ValueError("path_or_buf must be a file path if x is dask-backed")

    # Define constructor and indices
    # This paragraph was been copied from xarray.DataArray.to_pandas()
    constructors = {1: pd.Series,
                    2: pd.DataFrame,
                    3: pd.Panel}
    try:
        constructor = constructors[x.ndim]
    except KeyError:
        raise ValueError('cannot convert arrays with %s dimensions into '
                         'pandas objects' % x.ndim)
    indices = [x.get_index(dim) for dim in x.dims]

    # Merge chunks on all dimensions beyond the first
    x = x.chunk({d: -1 for d in x.dims[1:]})

    index = xarray.DataArray(indices[0]).chunk((x.chunks[0], ))

    tok = tokenize(x, index, indices[1:])
    name1 = 'to_csv_encode-' + tok
    name2 = 'to_csv_write-' + tok
    name3 = 'to_csv-' + tok

    dsk = {}

    for i in range(len(x.chunks[0])):
        xi = (x.name, i) + (0, ) * (x.ndim - 1)
        idx_i = (index.name, i)

        if 'header' in kwargs and i > 0:
            kwargs_i = kwargs.copy()
            del kwargs_i['header']
        else:
            kwargs_i = kwargs

        dsk[name1, i] = (
            kernels.to_csv, xi, constructor, [idx_i] + indices[1:], kwargs_i)

        if i == 0:
            dsk[name2, i] = (
                kernels.to_file, path_or_buf, 'bw', (name1, i))
        elif i < len(x.chunks[0]) - 1:
            dsk[name2, i] = (
                kernels.to_file, path_or_buf, 'ba', (name1, i), (name1, i - 1))
        else:
            dsk[name3] = (
                kernels.to_file, path_or_buf, 'ba', (name1, i), (name1, i - 1))

    return Delayed(name3, sharedict.merge(dsk, x, index))
