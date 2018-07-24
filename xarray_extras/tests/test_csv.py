import bz2
import gzip
import lzma
import pickle
import tempfile

import dask
import numpy as np
import pandas
import pytest
import xarray

from xarray_extras.csv import to_csv


def assert_to_csv(x, chunks, nogil, dtype, open_func=open, float_format='%f',
                  **kwargs):
    x = x.astype(dtype)
    if chunks:
        x = x.chunk(chunks)

    with tempfile.TemporaryDirectory() as tmp:
        x.to_pandas().to_csv(tmp + '/1.csv', float_format=float_format,
                             **kwargs)
        f = to_csv(x, tmp + '/2.csv', nogil=nogil, float_format=float_format,
                   **kwargs)
        dask.compute(f)

        with open_func(tmp + '/1.csv', 'rb') as fh:
            d1 = fh.read()
        with open_func(tmp + '/2.csv', 'rb') as fh:
            d2 = fh.read()
        assert d1 == d2


@pytest.mark.parametrize('dtype', [np.int64, np.float64])
@pytest.mark.parametrize('nogil', [False, True])
@pytest.mark.parametrize('chunks', [None, 1])
def test_series(chunks, nogil, dtype):
    x = xarray.DataArray(
        [1, 2, 3, 4],
        dims=['x'],
        coords={'x': [10, 20, 30, 40]})
    assert_to_csv(x, chunks, nogil, dtype)


@pytest.mark.parametrize('dtype', [np.int64, np.float64])
@pytest.mark.parametrize('nogil', [False, True])
@pytest.mark.parametrize('chunks', [None, 1])
def test_dataframe(chunks, nogil, dtype):
    x = xarray.DataArray(
        [[1, 2, 3, 4],
         [5, 6, 7, 8]],
        dims=['r', 'c'],
        coords={'r': ['a', 'b'], 'c': [10, 20, 30, 40]})
    assert_to_csv(x, chunks, nogil, dtype)


@pytest.mark.parametrize('dtype', [np.int64, np.float64])
@pytest.mark.parametrize('nogil', [False, True])
@pytest.mark.parametrize('chunks', [None, 1])
def test_multiindex(chunks, nogil, dtype):
    x = xarray.DataArray(
        [[1, 2],
         [3, 4]],
        dims=['r', 'c'],
        coords={'r1': ('r', ['r11', 'r12']),
                'r2': ('r', ['r21', 'r22']),
                'c1': ('c', ['c11', 'c12']),
                'c2': ('c', ['c21', 'c22'])})
    x = x.set_index(r=['r1', 'r2'], c=['c1', 'c2'])
    assert_to_csv(x, chunks, nogil, dtype)


@pytest.mark.parametrize('dtype', [np.int64, np.float64])
@pytest.mark.parametrize('nogil', [False, True])
@pytest.mark.parametrize('chunks', [None, 1])
def test_no_header(chunks, nogil, dtype):
    x = xarray.DataArray([[1, 2], [3, 4]])
    assert_to_csv(x, chunks, nogil, dtype, index=False, header=False)


@pytest.mark.parametrize('dtype', [np.int64, np.float64])
@pytest.mark.parametrize('nogil', [False, True])
@pytest.mark.parametrize('chunks', [None, 1])
def test_custom_header(chunks, nogil, dtype):
    x = xarray.DataArray([[1, 2], [3, 4]])
    assert_to_csv(x, chunks, nogil, dtype, header=['foo', 'bar'])


@pytest.mark.parametrize('encoding', ['utf-8', 'utf-16'])
@pytest.mark.parametrize('dtype', [np.int64, np.float64])
@pytest.mark.parametrize('nogil', [False, True])
@pytest.mark.parametrize('chunks', [None, 1])
def test_encoding(chunks, nogil, dtype, encoding):
    # Note: in Python 2.7, default encoding is ascii in pandas and utf-8 in
    # xarray_extras. Therefore we will not test the default.
    x = xarray.DataArray([[1], [2]], dims=['r', 'c'],
                         coords={'r': ['crème', 'foo'], 'c': ['brûlée']})
    assert_to_csv(x, chunks, nogil, dtype, encoding=encoding)


@pytest.mark.parametrize('sep', [',', '|'])
@pytest.mark.parametrize('float_format', ['%f', '%.2f', '%.15f', '%.5e'])
@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize('nogil', [False, True])
@pytest.mark.parametrize('chunks', [None, 1])
def test_kwargs(chunks, nogil, dtype, float_format, sep):
    x = xarray.DataArray([1.0, 1.1, 1.000000000000001, 123.456789])
    assert_to_csv(x, chunks, nogil, dtype, float_format=float_format, sep=sep)


@pytest.mark.parametrize('na_rep', ['', 'nan'])
@pytest.mark.parametrize('nogil', [False, True])
@pytest.mark.parametrize('chunks', [None, 1])
def test_na_rep(chunks, nogil, na_rep):
    x = xarray.DataArray([np.nan, 1])
    assert_to_csv(x, chunks, nogil, np.float64, na_rep=na_rep)


@pytest.mark.parametrize('compression,open_func', [
    ('gzip', gzip.open),
    ('bz2', bz2.open),
    ('xz', lzma.open),
])
@pytest.mark.parametrize('dtype', [np.int64, np.float64])
@pytest.mark.parametrize('nogil', [False, True])
@pytest.mark.parametrize('chunks', [None, 1])
def test_compression(chunks, nogil, dtype, compression, open_func):
    # Notes:
    # - compressed outputs won't be binary identical; only once uncompressed
    # - we are forcing the dask-based algorithm to compress two chunks
    if pandas.__version__ < '0.23':
        pytest.xfail("compression param requires pandas >=0.23")

    x = xarray.DataArray([1, 2])
    assert_to_csv(x, chunks, nogil, dtype, compression=compression,
                  open_func=open_func)


@pytest.mark.parametrize('dtype', [np.int64, np.float64])
@pytest.mark.parametrize('nogil', [False, True])
@pytest.mark.parametrize('chunks', [None, 1])
def test_empty(chunks, nogil, dtype):
    x = xarray.DataArray(
        [[1, 2, 3, 4]],
        dims=['r', 'c'],
        coords={'c': [10, 20, 30, 40]})
    x = x.isel(r=slice(0))
    assert_to_csv(x, chunks, nogil, dtype)


@pytest.mark.parametrize('x', [0, -2**63])
@pytest.mark.parametrize('index', ['a', 'a' * 1000])
@pytest.mark.parametrize('nogil', [False, True])
@pytest.mark.parametrize('chunks', [None, 1])
def test_buffer_overflow_int(chunks, nogil, index, x):
    a = xarray.DataArray([x], dims=['x'], coords={'x': [index]})
    assert_to_csv(a, chunks, nogil, np.int64)


@pytest.mark.parametrize('x', [0, np.nan, 1.000000000000001,
                               1.7901234406790122e+308])
@pytest.mark.parametrize('index,coord',
                         [(False, ''), (True, 'a'), (True, 'a' * 1000)])
@pytest.mark.parametrize('na_rep', ['', 'na' * 500])
@pytest.mark.parametrize('float_format',
                         ['%.16f', '%.1000f', 'a' * 1000 + '%.0f'])
@pytest.mark.parametrize('nogil', [False, True])
@pytest.mark.parametrize('chunks', [None, 1])
def test_buffer_overflow_float(chunks, nogil, float_format, na_rep, index,
                               coord, x):

    if nogil and not index and np.isnan(x) and na_rep == '':
        # Expected: b'""\n'
        # Actual: b'\n'
        pytest.xfail("pandas prints useless "" for empty lines")

    a = xarray.DataArray([x], dims=['x'], coords={'x': [coord]})
    assert_to_csv(a, chunks, nogil, np.float64, float_format=float_format,
                  na_rep=na_rep, index=index)


@pytest.mark.parametrize('encoding', ['utf-8', 'utf-16'])
@pytest.mark.parametrize('dtype', [str, object])
@pytest.mark.parametrize('chunks', [None, 1])
def test_pandas_only(chunks, dtype, encoding):
    x = xarray.DataArray(['foo', 'Crème brûlée'])
    assert_to_csv(x, chunks, False, dtype, encoding=encoding)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('chunks', [None, 1])
def test_pandas_only_complex(chunks, dtype):
    x = xarray.DataArray([1 + 2j])
    assert_to_csv(x, chunks, False, dtype)


@pytest.mark.parametrize('nogil', [False, True])
@pytest.mark.parametrize('chunks', [None, 1])
def test_mode(chunks, nogil):
    x = xarray.DataArray([1, 2])
    y = xarray.DataArray([3, 4])
    if chunks:
        x = x.chunk(chunks)
        y = y.chunk(chunks)

    with tempfile.TemporaryDirectory() as tmp:
        f = to_csv(x, tmp + '/1.csv', mode='a', nogil=nogil, index=False)
        dask.compute(f)
        f = to_csv(y, tmp + '/1.csv', mode='a', nogil=nogil, index=False)
        dask.compute(f)
        with open(tmp + '/1.csv') as fh:
            assert '1\n2\n3\n4\n' == fh.read()

        f = to_csv(y, tmp + '/1.csv', mode='w', nogil=nogil, index=False)
        dask.compute(f)
        with open(tmp + '/1.csv') as fh:
            assert '3\n4\n' == fh.read()


def test_none_fmt():
    """float_format=None differs between C and pandas; can't use assert_to_csv
    """
    x = xarray.DataArray([1.0, 1.1, 1.000000000000001, 123.456789])
    y = x.astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        to_csv(x, tmp + '/1.csv')
        to_csv(y, tmp + '/2.csv')

        with open(tmp + '/1.csv') as fh:
            assert '0,1.0\n1,1.1\n2,1.0\n3,123.456789\n' == fh.read()
        with open(tmp + '/2.csv') as fh:
            assert '0,1.0\n1,1.1\n2,1.0\n3,123.456787\n' == fh.read()


def test_pickle():
    x = xarray.DataArray([1, 2])
    with tempfile.TemporaryDirectory() as tmp:
        x.to_pandas().to_csv(tmp + '/1.csv')
        d = to_csv(x.chunk(1), tmp + '/2.csv')
        d = pickle.loads(pickle.dumps(d))
        d.compute()

        with open(tmp + '/1.csv', 'rb') as fh:
            d1 = fh.read()
        with open(tmp + '/2.csv', 'rb') as fh:
            d2 = fh.read()
        print(d1)
        print(d2)
        assert d1 == d2
