import pytest
from xarray import DataArray
from xarray.testing import assert_equal
from xarray_extras.sort import topk, argtopk, take_along_dim


@pytest.mark.parametrize('use_dask', [False, True])
@pytest.mark.parametrize('split_every', [None, 2])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('func,k,expect', [
    (topk, 3, [[5, 4, 3], [8, 7, 2]]),
    (topk, -3, [[1, 2, 3], [0, 1, 2]]),
    (argtopk, 3, [[3, 1, 4], [2, 0, 4]]),
    (argtopk, -3, [[0, 2, 4], [3, 1, 4]]),
])
def test_topk_argtopk(use_dask, split_every, transpose,
                      func, k, expect):
    a = DataArray([[1, 4, 2, 5, 3],
                   [7, 1, 8, 0, 2]],
                  dims=['y', 'x'],
                  coords={'y': ['y1', 'y2'],
                          'x': ['x1', 'x2', 'x3', 'x4', 'x5']})

    if transpose:
        a = a.T
    if use_dask:
        a = a.chunk(1)

    expect = DataArray(expect, dims=['y', 'x'], coords={'y': ['y1', 'y2']})
    actual = func(a, k, 'x', split_every=split_every)
    if use_dask:
        assert actual.chunks
    assert_equal(expect, actual)


@pytest.mark.parametrize('ind_use_dask', [False, True])
@pytest.mark.parametrize('a_use_dask', [False, True])
def test_take_along_dim(a_use_dask, ind_use_dask):
    """ind.ndim < a.ndim after broadcast
    """
    a = DataArray([[[1, 2, 3],
                    [4, 5, 6]],
                   [[7, 8, 9],
                    [10, 11, 12]]],
                  dims=['z', 'y', 'x'],
                  coords={'z': ['z1', 'z2'],
                          'y': ['y1', 'y2'],
                          'x': ['x1', 'x2', 'x3']})
    ind = DataArray([[[1, 0],
                      [2, 1]],
                     [[2, 0],
                      [2, 2]]],
                    dims=['w', 'y', 'x'],
                    coords={'y': ['y1', 'y2']})

    expect = DataArray([[[[2, 1],
                          [3, 1]],
                         [[6, 5],
                          [6, 6]]],
                        [[[8, 7],
                          [9, 7]],
                         [[12, 11],
                          [12, 12]]]],
                       dims=['z', 'y', 'w', 'x'],
                       coords={'z': ['z1', 'z2'],
                               'y': ['y1', 'y2']})

    if a_use_dask:
        a = a.chunk(1)
    if ind_use_dask:
        ind = ind.chunk(1)

    actual = take_along_dim(a, ind, 'x')
    if a_use_dask or ind_use_dask:
        assert actual.chunks
    assert_equal(expect, actual.compute())


@pytest.mark.parametrize('ind_use_dask', [False, True])
@pytest.mark.parametrize('a_use_dask', [False, True])
def test_take_along_dim2(a_use_dask, ind_use_dask):
    """ind.ndim < a.ndim after broadcast
    """
    a = DataArray([1, 2, 3], dims=['x'])
    ind = DataArray([[1, 0],
                     [2, 1]],
                    dims=['x', 'y'])

    expect = DataArray([[2, 3],
                        [1, 2]],
                       dims=['y', 'x'])

    if a_use_dask:
        a = a.chunk(1)
    if ind_use_dask:
        ind = ind.chunk(1)

    actual = take_along_dim(a, ind, 'x')
    if a_use_dask or ind_use_dask:
        assert actual.chunks
    assert_equal(expect, actual.compute())
