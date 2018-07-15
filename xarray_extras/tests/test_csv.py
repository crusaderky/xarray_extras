import bz2
import gzip
import lzma
import pickle
import tempfile
import pandas
import pytest
import xarray

from xarray_extras.csv import to_csv


def assert_to_csv(x, chunks, open_func=open, **kwargs):
    with tempfile.TemporaryDirectory() as tmp:
        x.to_pandas().to_csv(tmp + '/1.csv', **kwargs)
        to_csv(x, tmp + '/2.csv', **kwargs)
        to_csv(x.chunk(chunks), tmp + '/3.csv', **kwargs).compute()

        with open_func(tmp + '/1.csv', 'rb') as fh:
            d1 = fh.read()
        with open_func(tmp + '/2.csv', 'rb') as fh:
            d2 = fh.read()
        with open_func(tmp + '/3.csv', 'rb') as fh:
            d3 = fh.read()
        print(d1)
        print(d2)
        print(d3)
        assert d1 == d2
        assert d1 == d3


def test_series():
    x = xarray.DataArray(
        [1, 2, 3, 4],
        dims=['x'],
        coords={'x': [10, 20, 30, 40]})
    assert_to_csv(x, 5)


def test_dataframe():
    x = xarray.DataArray(
        [[1, 2, 3, 4],
         [5, 6, 7, 8]],
        dims=['r', 'c'],
        coords={'r': ['a', 'b'], 'c': [10, 20, 30, 40]})
    assert_to_csv(x, 1)


def test_multiindex():
    x = xarray.DataArray(
        [[1, 2],
         [3, 4]],
        dims=['r', 'c'],
        coords={'r1': ('r', ['r11', 'r12']),
                'r2': ('r', ['r21', 'r22']),
                'c1': ('c', ['c11', 'c12']),
                'c2': ('c', ['c21', 'c22'])})
    x = x.set_index(r=['r1', 'r2'], c=['c1', 'c2'])
    assert_to_csv(x, 1)


def test_no_header():
    x = xarray.DataArray([[1, 2], [3, 4]])
    assert_to_csv(x, 1, index=False, header=False)


def test_custom_header():
    x = xarray.DataArray([[1, 2], [3, 4]])
    assert_to_csv(x, 1, header=['foo', 'bar'])


@pytest.mark.parametrize('encoding', ['utf-8', 'utf-16'])
def test_encoding(encoding):
    # Note: in Python 2.7, default encoding is ascii in pandas and utf-8 in
    # xarray_extras. Therefore we will not test the default.
    x = xarray.DataArray(['brûlée'], dims=['x'], coords={'x': ['crème']})
    assert_to_csv(x, 1, encoding=encoding)


@pytest.mark.parametrize('float_format', ['%.2f', '%.15f', '%.5e'])
def test_kwargs(float_format):
    x = xarray.DataArray([123.456789])
    assert_to_csv(x, 1, float_format=float_format)


@pytest.mark.parametrize('compression,open_func', [
    ('gzip', gzip.open),
    ('bz2', bz2.open),
    ('xz', lzma.open),
])
def test_compression(compression, open_func):
    # Notes:
    # - compressed outputs won't be binary identical; only once uncompressed
    # - we are forcing the dask-based algorithm to compress two chunks
    if pandas.__version__ < '0.23':
        pytest.xfail("compression param requires pandas >=0.23")

    x = xarray.DataArray([1, 2])
    assert_to_csv(x, 1, compression=compression, open_func=open_func)


def test_empty():
    x = xarray.DataArray(
        [[1, 2, 3, 4]],
        dims=['r', 'c'],
        coords={'c': [10, 20, 30, 40]})
    x = x.isel(r=slice(0))
    assert_to_csv(x, 1)


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
