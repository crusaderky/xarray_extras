"""dask kernels for :mod:`xarray_extras.csv`
"""
import pandas
from .np_to_csv_py import snprintcsvd, snprintcsvi


def to_csv(x, index, columns, nogil, kwargs):
    """Format x into CSV and encode it to binary

    .. warning::
       This function does not release the GIL!

    :param x:
        numpy.ndarray with 1 or 2 dimensions
    :param index:
        row index
    :param columns:
        column index. None for Series or for DataFrame chunks beyond the first.
    :param bool nogil:
        If True, use accelerated C implementation. Several kwargs won't be
        processed correctly. If False, use pandas to_csv method (slow, and does
        not release the GIL).
    :param kwargs:
        arguments passed to pandas to_csv methods
    :returns:
        bytes
    """
    if x.ndim == 1:
        assert columns is None
        x_pd = pandas.Series(x, index)
    elif x.ndim == 2:
        x_pd = pandas.DataFrame(x, index, columns)
    else:
        assert False  # proper ValueError already raised in wrapper

    encoding = kwargs.pop('encoding', 'utf-8')
    if not nogil or not x.size:
        out = x_pd.to_csv(**kwargs)
        return out.encode(encoding)

    sep = kwargs.get('sep', ',')
    fmt = kwargs.get('float_format', None)
    na_rep = kwargs.get('na_rep', '')

    # Use pandas to format index
    if x.ndim == 1:
        x_df = x_pd.to_frame()
    else:
        x_df = x_pd
    kwargs_index = kwargs.copy()
    kwargs_index['header'] = False
    index_csv = x_df.iloc[:, :0].to_csv(**kwargs_index)
    index_csv = index_csv.strip().split('\n')
    if len(index_csv) != x.shape[0]:
        index_csv = '\n' * x.shape[0]
    else:
        index_csv = '\n'.join(
            f'{r}{sep}' if r else '' for r in index_csv) + '\n'

    # Invoke C code to format the values. This releases the GIL.
    if x.dtype.kind == 'i':
        body_csv = snprintcsvi(x, index_csv, sep)
    elif x.dtype.kind == 'f':
        body_csv = snprintcsvd(x, index_csv, sep, fmt, na_rep)
    else:
        raise ValueError("only int and float are supported")

    if x.ndim == 2 and kwargs.get('header') is not False:
        # Use pandas to format columns
        header_csv = x_df.iloc[:0, :].to_csv(**kwargs).encode('utf-8')
        body_csv = header_csv + body_csv

    if encoding not in {'ascii', 'utf-8'}:
        body_csv = body_csv.decode('utf-8').encode(encoding)
    return body_csv


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
