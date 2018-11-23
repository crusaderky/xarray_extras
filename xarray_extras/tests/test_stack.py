import numpy
import pandas
import pytest
import xarray
from xarray_extras.stack import proper_unstack


def test_proper_unstack():
    index = [
        ['x1', 'first'],
        ['x1', 'second'],
        ['x1', 'third'],
        ['x1', 'fourth'],
        ['x0', 'first'],
        ['x0', 'second'],
        ['x0', 'third'],
        ['x0', 'fourth'],
    ]
    index = pandas.MultiIndex.from_tuples(index, names=['x', 'count'])
    s = pandas.Series(list(range(8)), index)
    xa = xarray.DataArray(s)
    a = proper_unstack(xa, 'dim_0')
    b = xarray.DataArray(
        [[0, 1, 2, 3], [4, 5, 6, 7]],
        dims=['x', 'count'],
        coords={'x': ['x1', 'x0'],
                'count': ['first', 'second', 'third', 'fourth']})
    xarray.testing.assert_equal(a, b)
    with pytest.raises(AssertionError):
        xarray.testing.assert_equal(a, xa.unstack('dim_0'))
    for c in a.coords:
        assert a.coords[c].dtype.kind == 'U'


def test_proper_unstack_int_coords():
    index = [
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    index = pandas.MultiIndex.from_tuples(index, names=['x', 'count'])
    s = pandas.Series(list(range(8)), index)
    xa = xarray.DataArray(s)
    a = proper_unstack(xa, 'dim_0')
    b = xarray.DataArray([[0, 1, 2, 3], [4, 5, 6, 7]],
                         dims=['x', 'count'],
                         coords={'x': [1, 0],
                                 'count': [1, 2, 3, 4]})
    xarray.testing.assert_equal(a, b)
    with pytest.raises(AssertionError):
        xarray.testing.assert_equal(a, xa.unstack('dim_0'))
    for c in a.coords:
        assert a.coords[c].dtype.kind == 'i'


def test_proper_unstack_mixed_coords():
    index = [
        [1, 1],
        [1, 2.2],
        [1, '3'],
        [1, 'fourth'],
        ['x0', 1],
        ['x0', 2.2],
        ['x0', '3'],
        ['x0', 'fourth'],
    ]
    index = pandas.MultiIndex.from_tuples(index, names=['x', 'count'])
    s = pandas.Series(list(range(8)), index)
    xa = xarray.DataArray(s)
    a = proper_unstack(xa, 'dim_0')
    b = xarray.DataArray([[0, 1, 2, 3], [4, 5, 6, 7]],
                         dims=['x', 'count'],
                         coords={'x': [1, 'x0'],
                                 'count': [1, 2.2, '3', 'fourth']})
    xarray.testing.assert_equal(a, b)
    for c in a.coords:
        assert a.coords[c].dtype.kind == 'U'


def test_proper_unstack_dataset():
    a = xarray.DataArray(
        [[1, 2, 3, 4],
         [5, 6, 7, 8]],
        dims=['x', 'col'],
        coords={'x': ['x0', 'x1'],
                'col': pandas.MultiIndex.from_tuples([('u0', 'v0'),
                                                      ('u0', 'v1'),
                                                      ('u1', 'v0'),
                                                      ('u1', 'v1')],
                                                     names=['u', 'v'])})
    xa = xarray.Dataset({'foo': a, 'bar': ('w', [1, 2]), 'baz': numpy.pi})
    b = proper_unstack(xa, 'col')
    c = xarray.DataArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                         dims=['x', 'u', 'v'],
                         coords={'x': ['x0', 'x1'],
                                 'u': ['u0', 'u1'],
                                 'v': ['v0', 'v1']})
    d = xarray.Dataset({'foo': c, 'bar': ('w', [1, 2]), 'baz': numpy.pi})
    assert b.equals(d)
    for c in b.coords:
        assert b.coords[c].dtype.kind == 'U'


def test_proper_unstack_other_mi():
    a = xarray.DataArray(
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [1, 2, 3, 4],
         [5, 6, 7, 8]],
        dims=['row', 'col'],
        coords={'row': pandas.MultiIndex.from_tuples([('x0', 'w0'),
                                                      ('x0', 'w1'),
                                                      ('x1', 'w0'),
                                                      ('x1', 'w1')],
                                                     names=['x', 'w']),
                'col': pandas.MultiIndex.from_tuples([('y0', 'z0'),
                                                      ('y0', 'z1'),
                                                      ('y1', 'z0'),
                                                      ('y1', 'z1')],
                                                     names=['y', 'z'])})
    b = proper_unstack(a, 'row')
    c = xarray.DataArray(
        [[[1, 5], [1, 5]],
         [[2, 6], [2, 6]],
         [[3, 7], [3, 7]],
         [[4, 8], [4, 8]]],
        dims=['col', 'x', 'w'],
        coords={'col': pandas.MultiIndex.from_tuples([('y0', 'z0'),
                                                      ('y0', 'z1'),
                                                      ('y1', 'z0'),
                                                      ('y1', 'z1')],
                                                     names=['y', 'z']),
                'x': ['x0', 'x1'],
                'w': ['w0', 'w1']})
    xarray.testing.assert_equal(b, c)
