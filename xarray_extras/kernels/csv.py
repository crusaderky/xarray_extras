"""dask kernels for :mod:`xarray_extras.csv
"""
from threading import Lock
import pandas


# Wrap pandas.DataFrame.to_csv in a lock to limit
# performance degradation due to GIL contention
to_csv_lock = Lock()


def to_csv(x, index, columns, kwargs):
    """Format x into CSV and encode it to binary

    .. warning::
       This function does not release the GIL!

    :param x:
        numpy.ndarray with 1 or 2 dimensions
    :param index:
        row index
    :param columns:
        column index. None for Series or for DataFrame chunks beyond the first.
    :param kwargs:
        arguments passed to pandas to_csv methods
    :returns:
        bytes
    """
    if x.ndim == 1:
        assert columns is None
        x = pandas.Series(x, index)
    elif x.ndim == 2:
        x = pandas.DataFrame(x, index, columns)
    else:
        assert False  # proper ValueError already raised in wrapper

    encoding = kwargs.pop('encoding', 'utf-8')
    with to_csv_lock:
        out = x.to_csv(**kwargs)
    return out.encode(encoding)


def to_file(fname, mode, data, rr_token=None):
    """Write data to file

    :param fname:
        File path on disk
    :param mode:
        As in 'open'
    :param data:
        Binary or text data to write
    :param rr_token:
        Round-robin token passed by to_file from the previous chunk. It
        guarantees write order across multiple, otherwise parallel, tasks. This
        parameter is only used by the dask scheduler.
    """
    with open(fname, mode) as fh:
        fh.write(data)
