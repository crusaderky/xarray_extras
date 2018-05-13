import numpy
import xarray
import pytest
from xarray.testing import assert_equal
import xarray_extras.cumulatives as cum


# Skip 0 and 1 as they're neutral in addition and multiplication
INPUT = xarray.DataArray(
    [[2, 20, 25],
     [3, 30, 35],
     [4, 40, 45],
     [5, 50, 55]],
    dims=['t', 's'],
    coords={
        't': numpy.array(
            ['1990-12-30',
             '2000-12-30',
             '2005-12-30',
             '2010-12-30'], dtype='<M8[D]'),
        's': ['s1', 's2', 's3']})


T_COMPOUND_MATRIX = xarray.DataArray(
    numpy.array([
        ['1990-12-30', 'NaT', 'NaT'],
        ['1990-12-30', '2005-12-30', 'NaT'],
        ['2000-12-30', '1990-12-30', 'NaT'],
        ['2010-12-30', '1990-12-30', '2005-12-30']], dtype='<M8[D]'),
    dims=['t2', 'c'],
    coords={'t2': [10, 20, 30, 40]})


S_COMPOUND_MATRIX = xarray.DataArray(
    [['s3', 's2'],
     ['s1', '']],
    dims=['s2', 'c'],
    coords={'s2': ['foo', 'bar']})

DTYPES = (
    # There's a bug in numba.guvectorize for u8, u16, u32
    # i8 and i16 are too short to store the output
    'int32', 'int64',
    'uint64',
    'float32', 'float64',
    'complex64', 'complex128'
)


@pytest.mark.parametrize('func, meth', [
    (cum.compound_sum, 'sum'),
    (cum.compound_prod, 'prod'),
    (cum.compound_mean, 'mean')])
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('use_dask', [False, True])
def test_compound_t(func, meth, dtype, use_dask):
    x = INPUT.astype(dtype)
    c = T_COMPOUND_MATRIX
    expect = xarray.concat([
        getattr(x.isel(t=[0]), meth)('t'),
        getattr(x.isel(t=[0, 2]), meth)('t'),
        getattr(x.isel(t=[1, 0]), meth)('t'),
        getattr(x.isel(t=[3, 0, 2]), meth)('t'),
    ], dim='t2').T.astype(dtype)
    expect.coords['t2'] = c.coords['t2']

    if use_dask:
        x = x.chunk({'s': 2})
        expect = expect.chunk({'s': 2})
        c = c.chunk()

    actual = func(x, c, 't', 'c')

    if use_dask:
        assert_equal(expect.compute(), actual.compute())
    else:
        assert_equal(expect, actual)

    assert expect.dtype == actual.dtype
    assert actual.chunks == expect.chunks


@pytest.mark.parametrize('func, meth', [
    (cum.compound_sum, 'sum'),
    (cum.compound_prod, 'prod'),
    (cum.compound_mean, 'mean')])
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('use_dask', [False, True])
def test_compound_s(func, meth, dtype, use_dask):
    x = INPUT.astype(dtype)
    c = S_COMPOUND_MATRIX
    expect = xarray.concat([
        getattr(x.sel(s=['s3', 's2']), meth)('s'),
        getattr(x.sel(s=['s1']), meth)('s'),
    ], dim='s2').T.astype(dtype)
    expect.coords['s2'] = c.coords['s2']

    if use_dask:
        x = x.chunk({'t': 2})
        expect = expect.chunk({'t': 2})
        c = c.chunk()

    actual = func(x, c, 's', 'c')

    if use_dask:
        assert_equal(expect.compute(), actual.compute())
    else:
        assert_equal(expect, actual)

    assert expect.dtype == actual.dtype
    assert actual.chunks == expect.chunks


@pytest.mark.parametrize('dtype', [float, int, 'complex128'])
@pytest.mark.parametrize('skipna', [False, True, None])
@pytest.mark.parametrize('use_dask', [False, True])
def test_cummean(use_dask, skipna, dtype):
    x = INPUT.copy(deep=True).astype(dtype)
    if dtype in (float, 'complex128'):
        x[2, 1] = numpy.nan

    expect = xarray.concat([
        x[:1].mean('t', skipna=skipna),
        x[:2].mean('t', skipna=skipna),
        x[:3].mean('t', skipna=skipna),
        x[:4].mean('t', skipna=skipna),
    ], dim='t')
    expect.coords['t'] = x.coords['t']
    if use_dask:
        x = x.chunk({'s': 2, 't': 3})
        expect = expect.chunk({'s': 2, 't': 3})

    actual = cum.cummean(x, 't', skipna=skipna)
    assert_equal(expect, actual)
    assert expect.dtype == actual.dtype
    assert actual.chunks == expect.chunks
