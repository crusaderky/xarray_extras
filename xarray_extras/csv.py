"""CSV file type support
"""
import pandas
import xarray
import dask.array as da
from dask.base import tokenize
from dask.delayed import Delayed
from dask import sharedict
from .kernels import csv as kernels


__all__ = ('to_csv', )


def to_csv(x, path_or_buf, **kwargs):
    """Print DataArray to CSV.

    When x has numpy backend, this function is equivalent to::

        x.to_pandas().to_csv(path_or_buf, **kwargs)

    When x has dask backend, this function returns a dask delayed object which
    will write to the disk only when its .compute() method is invoked.

    Formatting and optional compression are parallelised across all available
    CPUs, using one dask task per chunk on the first dimension. Chunks on other
    dimensions will be merged ahead of computation.

    :param x:
        xarray.DataArray with one or two dimensions
    :param path_or_buf:
        File path or file-like object
    :param kwargs:
        Passed verbatim to :meth:`pandas.DataFrame.to_csv` or
        :meth:`pandas.Series.to_csv`

    **Limitations**

    - When x has dask backend, path_or_buf must be a file path. Fancy URIs are
      not (yet) supported.
    - When x has dask backend, compression='zip' is not supported. All other
      compression methods (gzip, bz2, xz) are supported.

    **Distributed**

    This function supports `dask distributed
    <https://distributed.readthedocs.io/>`_, with the caveat that all workers
    must write to the same shared mountpoint and that the shared filesystem
    must strictly guarantee **close-open coherency**, meaning that one must be
    able to call write() and then close() on a file descriptor from one host
    and then immediately afterwards open() from another host and see the output
    from the first host. Note that, for performance reasons, most network
    filesystems do not enable this feature by default.

    Alternatively, one may write to local mountpoints and then manually collect
    and concatenate the partial outputs.
    """
    if not isinstance(x, xarray.DataArray):
        raise ValueError("first argument must be a DataArray")

    # Fast exit for numpy backend
    if not x.chunks:
        x.to_pandas().to_csv(path_or_buf, ** kwargs)
        return None

    # Health checks
    if not isinstance(path_or_buf, str):
        raise ValueError("path_or_buf must be a file path if x is dask-backed")

    if x.ndim not in (1, 2):
        raise ValueError('cannot convert arrays with %s dimensions into '
                         'pandas objects' % x.ndim)

    # Define compress function
    compression = kwargs.pop('compression', None)
    if compression is None:
        compress = None
    elif compression == 'gzip':
        import gzip
        compress = gzip.compress
    elif compression == 'bz2':
        import bz2
        compress = bz2.compress
    elif compression == 'xz':
        import lzma
        compress = lzma.compress
    elif compression == 'zip':
        raise NotImplementedError("zip compression is not supported when"
                                  "data has dask backend")
    else:
        raise ValueError("Unrecognized compression: %s" % compression)

    # Merge chunks on all dimensions beyond the first
    x = x.chunk((x.chunks[0],) + tuple((s, ) for s in x.shape[1:]))

    # Extract row and columns indices
    indices = [x.get_index(dim) for dim in x.dims]
    if x.ndim == 2:
        index, columns = indices
    else:
        index = indices[0]
        columns = None

    # Convert row index to dask. Do not use DataArray(indices[0]).chunk(), as
    # it will cause the token to become unstable
    if isinstance(index, pandas.MultiIndex):
        index_name = tuple(index.names)
    else:
        index_name = index.name
    index = da.from_array(index, chunks=(x.chunks[0], ))

    # Manually define the dask graph
    tok = tokenize(x.data, index, columns, compression, path_or_buf, kwargs)
    name1 = 'to_csv_encode-' + tok
    name2 = 'to_csv_write-' + tok
    name3 = 'to_csv-' + tok

    dsk = {}

    assert x.chunks[0]
    for i in range(len(x.chunks[0])):
        x_i = (x.data.name, i) + (0, ) * (x.ndim - 1)
        idx_i = (index.name, i)

        if i == 0:
            # First chunk. Overwrite file if it already exists; print header
            dsk[name1, i] = (kernels.to_csv, x_i, index_name, idx_i, columns,
                             compress, kwargs)
            dsk[name2, i] = (kernels.to_file, path_or_buf, 'bw', (name1, i))
        else:
            kwargs_i = kwargs.copy()
            kwargs_i['header'] = False
            dsk[name1, i] = (kernels.to_csv, x_i, index_name, idx_i, columns,
                             compress, kwargs_i)
            dsk[name2, i] = (kernels.to_file, path_or_buf, 'ba', (name1, i),
                             (name2, i - 1))

    # Rename final key
    dsk[name3] = dsk.pop((name2, i))

    return Delayed(name3, sharedict.merge(
        dsk, x.__dask_graph__(), index.__dask_graph__()))
