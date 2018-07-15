import dask.array as da
import numpy as np
import pytest
from xarray import DataArray, Dataset, apply_ufunc
from xarray.testing import assert_equal, assert_allclose
from xarray_extras.interpolate import splrep, splev
import xarray_extras.kernels.interpolate as kernels


@pytest.mark.parametrize('k,expect', [
    (0, [40, 55, 4]),
    (1, [45, 35, 4.5]),
    (2, [45, 37.90073529, np.nan]),
    (3, [45, 39.69583333, np.nan])])
def test_0d(k, expect):
    """
    - Test different orders
    - Test unsorted x
    - Test what happens when a series contains NaN
    """
    y = DataArray([[10, 20, 30, 40, 50, 60],
                   [11, 28, 39, 55, 15, -2],
                   [np.nan, 2, 3, 4, 5, 6]],
                  dims=['y', 'x'],
                  coords={'x': [1, 2, 3, 4, 5, 6],
                          'y': ['y1', 'y2', 'y3']})
    tck = splrep(y, 'x', k)
    expect = DataArray(expect, dims=['y'],
                       coords={'x': 4.5,
                               'y': ['y1', 'y2', 'y3']}).astype(float)
    assert_allclose(splev(4.5, tck), expect, rtol=0, atol=1e-6)


@pytest.mark.parametrize('x_new,expect', [
    # list
    ([3.5, 4.5],
     DataArray([35., 45.], dims=['x'], coords={'x': [3.5, 4.5]})),
    # tuple
    ((3.5, 4.5),
     DataArray([35., 45.], dims=['x'], coords={'x': [3.5, 4.5]})),
    # np.array
    (np.array([3.5, 4.5]),
     DataArray([35., 45.], dims=['x'], coords={'x': [3.5, 4.5]})),
    # da.Array
    (da.from_array(np.array([3.5, 4.5]), chunks=1),
     DataArray([35., 45.], dims=['x'], coords={'x': [3.5, 4.5]}).chunk(1)),
    # DataArray, same dim as y, no coord
    (DataArray([3.5, 4.5], dims=['x']),
     DataArray([35., 45.], dims=['x'])),
    # DataArray, same dim as y, with coord
    (DataArray([3.5, 4.5], dims=['x'], coords={'x': [100, 200]}),
     DataArray([35., 45.], dims=['x'], coords={'x': [100, 200]})),
    # DataArray, different dim as y
    (DataArray([3.5, 4.5], dims=['t']),
     DataArray([35., 45.], dims=['t']))
])
def test_1d(x_new, expect):
    """
    - Test 1d case
    - Test auto-casting of various types of x_new
    """
    y = DataArray([10, 20, 30, 40, 50, 60],
                  dims=['x'], coords={'x': [1, 2, 3, 4, 5, 6]})
    tck = splrep(y, 'x', k=1)
    y_new = splev(x_new, tck)
    assert_equal(y_new, expect)
    assert y_new.chunks == expect.chunks


@pytest.mark.parametrize(
    'chunk_y,chunk_x_new,expect_chunks_tck,expect_chunks_y_new', [
        (False, False, {}, None),
        (False, True, {}, ((1, 1), (1, 1), (2,))),
        (True, False,
         {'y': (1, 1), 'x': (6,)},
         ((2,), (2,), (1, 1))),
        (True, True,
         {'y': (1, 1), 'x': (6,)},
         ((1, 1), (1, 1), (1, 1))),
    ])
def test_nd(chunk_y, chunk_x_new, expect_chunks_tck, expect_chunks_y_new):
    """
    - Test ND y vs. ND x_new
    - Test dask
    """
    y = DataArray([[10, 20, 30, 40, 50, 60],
                   [11, 28, 39, 55, 15, -2]],
                  dims=['y', 'x'],
                  coords={'x': [1, 2, 3, 4, 5, 6],
                          'y': ['y1', 'y2']})
    x_new = DataArray([[3.5, 4.5],
                       [1.5, 5.5]],
                      dims=['w', 'z'],
                      coords={'w': [100, 200],
                              'z': ['foo', 'bar']})
    expect_tck = Dataset(
        data_vars={
            't': ('__t__', [1., 1., 1., 1., 3., 4., 6., 6., 6., 6.]),
            'c': (('x', 'y'),
                  [[10., 11.],
                   [16.666667, 33.118519],
                   [26.666667, 20.762963],
                   [43.333333, 84.003704],
                   [53.333333, -25.251852],
                   [60., -2]]),
        },
        coords={'x': [1, 2, 3, 4, 5, 6],
                'y': ['y1', 'y2']},
        attrs={'spline_dim': 'x',
               'k': 3})

    expect_y_new = DataArray(
        [[[35., 51.0375],
          [45., 39.69583333]],
         [[15., 22.72083333],
          [55., -3.945833]]],
        dims=['w', 'z', 'y'],
        coords={
            'w': [100, 200],
            'y': ['y1', 'y2'],
            'z': ['foo', 'bar'],
        })

    if chunk_y:
        y = y.chunk({'y': 1})
    if chunk_x_new:
        x_new = x_new.chunk(1)

    tck = splrep(y, 'x', k=3)
    assert_allclose(tck.compute(), expect_tck, atol=1e-6, rtol=0)
    y_new = splev(x_new, tck)
    assert_allclose(y_new.compute(), expect_y_new, atol=1e-6, rtol=0)

    assert tck.chunks == expect_chunks_tck
    assert y_new.chunks == expect_chunks_y_new


@pytest.mark.parametrize('contiguous', [False, True])
@pytest.mark.parametrize('transpose', [False, True])
def test_transpose(transpose, contiguous):
    """Test that the interpolation dim does not need to be on axis 0
    """
    y = DataArray([[10, 20],
                   [30, 40],
                   [50, 60]],
                  dims=['y', 'x'],
                  coords={'x': [1, 2],
                          'y': ['y1', 'y2', 'y3']})
    expect = DataArray([[15., 35., 55.],
                        [10., 30., 50.]],
                       dims=['x', 'y'],
                       coords={'x': [1.5, 1.0],
                               'y': ['y1', 'y2', 'y3']})

    if transpose:
        y = y.T
    if contiguous:
        y = apply_ufunc(np.ascontiguousarray, y)

    tck = splrep(y, 'x', 1)
    y_new = splev([1.5, 1.0], tck)
    assert_equal(expect, y_new)


@pytest.mark.parametrize('x_new_dtype', [np.uint32, np.uint64, np.int32,
                                         np.int64, np.float32])
@pytest.mark.parametrize('x_dtype', [np.uint32, np.uint64, np.int32,
                                     np.int64, np.float32])
def test_nonfloat(x_dtype, x_new_dtype):
    """Test numeric x that isn't float64
    """
    x = np.array([0, 100])

    y = DataArray((x * 3).astype(x_dtype), dims=['x'],
                  coords={'x': x.astype(x_dtype)})
    x_new = np.array([50]).astype(x_new_dtype)
    expect = DataArray([150.], dims=['x'],
                       coords={'x': x_new})

    tck = splrep(y, 'x', k=1)
    y_new = splev(x_new, tck)
    assert_equal(expect, y_new)


@pytest.mark.parametrize('x_new_dtype', ['<M8[D]', '<M8[s]', '<M8[ns]'])
@pytest.mark.parametrize('x_dtype', ['<M8[D]', '<M8[s]', '<M8[ns]'])
def test_dates(x_dtype, x_new_dtype):
    """
    - Test mismatched date formats on x and x_new
    - Test clip extrapolation on test_dates
    """
    y = DataArray([10, 20], dims=['x'],
                  coords={'x': np.array(["2000-01-01", "2001-01-01"])
                  .astype(x_dtype)})
    x_new = np.array(["2000-04-20", "2002-07-28"]).astype(x_new_dtype)
    expect = DataArray([13.00546448, 20.], dims=['x'],
                       coords={'x': x_new.astype('<M8[ns]')})

    tck = splrep(y, 'x', k=1)
    y_new = splev(x_new, tck, extrapolate='clip')
    assert_allclose(expect, y_new, atol=1e-6, rtol=0)


@pytest.mark.parametrize('extrapolate,expect', [
    (True, [-1.36401507, -1.22694955]),
    (False, [np.nan, np.nan]),
    ('clip', [0, np.sin(9)]),
    ('periodic', [0.98935825, 0.84147098]),
])
def test_extrapolate(extrapolate, expect):
    """Test all possible extrapolate parameters
    """
    x = np.arange(10)
    y = DataArray(np.sin(x), dims=['x'], coords={'x': x})
    x_new = [-1, 10]
    expect = DataArray(expect, dims=['x'], coords={'x': x_new})

    tck = splrep(y, 'x', k=3)
    y_new = splev(x_new, tck, extrapolate=extrapolate)
    assert_allclose(expect, y_new, atol=1e-6, rtol=0)


def test_dim_collision():
    """y and x_new have overlapping dims besides the interpolation dim
    """
    y = DataArray([[10, 20],
                   [11, 28]],
                  dims=['y', 'x'],
                  coords={'x': [1, 2],
                          'y': ['y1', 'y2']})
    x_new = DataArray([1, 1], dims=['y'])
    tck = splrep(y, 'x', 1)
    with pytest.raises(ValueError) as excinfo:
        splev(x_new, tck)
    assert str(excinfo.value) == 'Overlapping dims between interpolated ' \
                                 'array and x_new: y'


def test_duplicates():
    """x contains non-unique points
    """
    y = DataArray([10, 20, 30], dims=['x'], coords={'x': [1, 1, 2]})
    with pytest.raises(ValueError) as excinfo:
        splrep(y, 'x', 1)
    assert str(excinfo.value) == "Expect x to be a 1-D sorted array-like."


def test_chunked_x():
    """x is chunked
    """
    y = DataArray([10, 20], dims=['x'], coords={'x': [1, 2]}).chunk(1)

    with pytest.raises(NotImplementedError) as excinfo:
        splrep(y, 'x', 1)
    assert str(excinfo.value) == 'Unsupported: multiple chunks on ' \
                                 'interpolation dim'


def test_distributed():
    def ro_array(a):
        a = np.array(a)
        a.setflags(write=False)
        # Return a view of a, so that setting the write flag on the view is not
        # enough
        return a[:]

    x = ro_array([1., 2.])
    y = ro_array([10., 20.])
    t = kernels.make_interp_knots(x, k=1)
    t = ro_array(t)
    c = kernels.make_interp_coeffs(x, y, k=1, t=t)
    c = ro_array(c)
    x_new = ro_array([1.5, 1.8])
    kernels.splev(x_new, t, c, k=1)
