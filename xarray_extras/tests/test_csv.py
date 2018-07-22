import bz2
import gzip
import lzma
import pickle
import tempfile

import numpy as np
import pandas
import pytest
import xarray

from xarray_extras.csv import to_csv


def assert_to_csv(x, chunks, nogil, dtype, open_func=open, float_format='%f', **kwargs):
    x = x.astype(dtype)
    with tempfile.TemporaryDirectory() as tmp:
        x.to_pandas().to_csv(tmp + '/1.csv', float_format=float_format, **kwargs)
        if chunks:
            x = x.chunk(chunks)
            to_csv(x, tmp + '/2.csv', nogil=nogil, float_format=float_format, **kwargs).compute()
        else:
            to_csv(x, tmp + '/2.csv', nogil=nogil, float_format=float_format, **kwargs)

        with open_func(tmp + '/1.csv', 'rb') as fh:
            d1 = fh.read()
        with open_func(tmp + '/2.csv', 'rb') as fh:
            d2 = fh.read()
        print()
        print(d1)
        print(d2)
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
    x = xarray.DataArray([[1]], dims=['r', 'c'],
                         coords={'r': ['crème'], 'c': ['brûlée']})
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


@pytest.mark.parametrize('encoding', ['utf-8', 'utf-16'])
@pytest.mark.parametrize('dtype', [str, object])
@pytest.mark.parametrize('chunks', [None, 1])
def test_pandas_only(chunks, dtype, encoding):
    x = xarray.DataArray(['foo', 'Crème brûlée'])
    assert_to_csv(x, chunks, False, dtype, encoding=encoding)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('chunks', [None, 1])
def test_pandas_only_complex(chunks, dtype):
    x = xarray.DataArray([1+2j])
    assert_to_csv(x, chunks, False, dtype)


def test_none_fmt():
    """float_format=None differs between C and pandas; can't use assert_to_csv
    """
    x = xarray.DataArray([1.0, 1.1, 1.000000000000001, 123.456789])
    y = x.astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        to_csv(x, tmp + '/1.csv')
        to_csv(y, tmp + '/2.csv')

        with open(tmp + '/1.csv', 'rb') as fh:
            d1 = fh.read()
        with open(tmp + '/2.csv', 'rb') as fh:
            d2 = fh.read()
        print()
        print(d1)
        print(d2)
        assert d1 == b'0,1.0\n1,1.1\n2,1.0\n3,123.456789\n'
        assert d2 == b'0,1.0\n1,1.1\n2,1.0\n3,123.456787\n'


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
