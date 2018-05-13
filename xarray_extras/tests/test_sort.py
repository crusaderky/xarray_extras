import pytest
from xarray import DataArray
from xarray.testing import assert_equal
from xarray_extras.sort import topk, argtopk


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
