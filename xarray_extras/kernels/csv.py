"""dask kernels for :mod:`xarray_extras.csv
"""
from multiprocessing import Process, Queue
import pandas


def to_csv(x, index_name, index, columns, compress, kwargs):
    """Format x into CSV, encode it, and optionally compress it.

    Since :meth:`pandas.DataFrame.to_csv` does not release the GIL, the actual
    computation is performed in a subprocess.

    :param x:
        numpy.ndarray with 1 or 2 dimensions
    :param index_name:
        index name, or tuple of index names in case of MultiIndex
    :param index:
        row index
    :param columns:
        column index. None for Series or for DataFrame chunks beyond the first.
    :param compress:
        callable to compress the data, e.g. :func:`gzip.compress`, or None
    :param kwargs:
        arguments passed to pandas to_csv methods
    """
    if isinstance(index_name, tuple):
        index = pandas.MultiIndex.from_tuples(index.tolist(), names=index_name)
    else:
        index = pandas.Index(index, name=index_name)

    if x.ndim == 1:
        assert columns is None
        x = pandas.Series(x, index)
    elif x.ndim == 2:
        x = pandas.DataFrame(x, index, columns)
    else:
        assert False  # proper ValueError already raised in wrapper

    queue = Queue()
    p = Process(target=to_csv_subprocess, args=(queue, x, compress, kwargs))
    p.start()
    out = queue.get()
    p.join()  # this blocks until the process terminates
    return out


def to_csv_subprocess(queue, x, compress, kwargs):
    """Helper function of :func:`to_csv`, running inside a subprocess

    :param queue:
        :class:`multiprocessing.Queue`
    :param x:
        :class:`pandas.DataFrame` or :class:`pandas.Series`
    :param compress:
        callable to compress the data, e.g. :func:`gzip.compress`, or None
    :param kwargs:
        arguments for pandas to_csv method
    :returns:
        (through queue): binary encoded, and possibly compressed, data
    """
    encoding = kwargs.pop('encoding', 'utf-8')
    out = x.to_csv(**kwargs)
    out = out.encode(encoding)
    if compress:
        out = compress(out)
    queue.put(out)


def to_file(fname, mode, data, rr_token=None):
    """Write data to file

    :param fname:
        File path on disk
    :param mode:
        As in 'open'
    :param data:
        Binary or text data to write
    :param rr_token:
        Round-robin token passed from the invocation of to_file from the
        previous chunk. It guarantees write order across multiple, otherwise
        parallel, tasks. This parameter is only used by the dask scheduler.
    """
    with open(fname, mode) as fh:
        fh.write(data)
