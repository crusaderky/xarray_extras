"""dask kernels for :mod:`xarray_extras.csv`
"""
import os
import pandas
from .np_to_csv_py import snprintcsvd, snprintcsvi
from ..backport.pandas import to_csv as pd_to_csv


def to_csv(x, index, columns, first_chunk, nogil, kwargs):
    """Format x into CSV and encode it to binary

    :param x:
        numpy.ndarray with 1 or 2 dimensions
    :param index:
        row index
    :param columns:
        column index. None for Series or for DataFrame chunks beyond the first.
    :param bool first_chunk:
        True if this is the first chunk; False otherwise
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
    header = kwargs.pop('header', True)
    line_terminator = kwargs.pop('line_terminator', os.linesep)

    if not nogil or not x.size:
        out = pd_to_csv(x_pd, header=header, line_terminator=line_terminator,
                        **kwargs)
        bout = out.encode(encoding)
        if encoding == 'utf-16' and not first_chunk:
            # utf-16 contains a bang at the beginning of the text. However,
            # when concatenating multiple chunks we don't want to replicate it.
            assert bout[:2] == b'\xff\xfe'
            bout = bout[2:]
        return bout

    sep = kwargs.get('sep', ',')
    fmt = kwargs.get('float_format', None)
    na_rep = kwargs.get('na_rep', '')

    # Use pandas to format index
    if x.ndim == 1:
        x_df = x_pd.to_frame()
    else:
        x_df = x_pd

    index_csv = pd_to_csv(x_df.iloc[:, :0], header=False, line_terminator='\n',
                          **kwargs)
    index_csv = index_csv.strip().split('\n')
    if len(index_csv) != x.shape[0]:
        index_csv = '\n' * x.shape[0]
    else:
        index_csv = '\n'.join(
            r + sep if r else '' for r in index_csv) + '\n'

    # Invoke C code to format the values. This releases the GIL.
    if x.dtype.kind == 'i':
        body_csv = snprintcsvi(x, index_csv, sep)
    elif x.dtype.kind == 'f':
        body_csv = snprintcsvd(x, index_csv, sep, fmt, na_rep)
    else:
        raise NotImplementedError("only int and float are supported when "
                                  "nogil=True")

    if header is not False:
        header_csv = pd_to_csv(x_df.iloc[:0, :], header=header,
                               line_terminator='\n', **kwargs).encode('utf-8')
        body_csv = header_csv + body_csv

    if encoding not in {'ascii', 'utf-8'}:
        # Everything is encoded in UTF-8 until this moment. Recode if needed.
        body_csv = body_csv.decode('utf-8')
        if line_terminator != '\n':
            body_csv = body_csv.replace('\n', line_terminator)
        body_csv = body_csv.encode(encoding)
        if encoding == 'utf-16' and not first_chunk:
            # utf-16 contains a bang at the beginning of the text. However,
            # when concatenating multiple chunks we don't want to replicate it.
            assert body_csv[:2] == b'\xff\xfe'
            body_csv = body_csv[2:]
    elif line_terminator != '\n':
        body_csv = body_csv.replace(b'\n', line_terminator.encode('utf-8'))

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
