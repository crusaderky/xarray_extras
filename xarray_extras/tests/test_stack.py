import numpy
import pandas
import pytest
import xarray
from xarray_extras.stack import proper_unstack


def test_proper_unstack_order():
    # Note: using MultiIndex.from_tuples is NOT the same thing as
    # round-tripping DataArray.stack().unstack(), as the latter is not
    # affected by the re-ordering issue
    index = pandas.MultiIndex.from_tuples(
        [['x1', 'first'],
         ['x1', 'second'],
         ['x1', 'third'],
         ['x1', 'fourth'],
         ['x0', 'first'],
         ['x0', 'second'],
         ['x0', 'third'],
         ['x0', 'fourth']],
        names=['x', 'count'])
    xa = xarray.DataArray(
        numpy.arange(8), dims=['dim_0'], coords={'dim_0': index})

    a = proper_unstack(xa, 'dim_0')
    b = xarray.DataArray(
        [[0, 1, 2, 3], [4, 5, 6, 7]],
        dims=['x', 'count'],
        coords={'x': ['x1', 'x0'],
                'count': ['first', 'second', 'third', 'fourth']})
    xarray.testing.assert_equal(a, b)
    with pytest.raises(AssertionError):
        # Order is different
        xarray.testing.assert_equal(a, xa.unstack('dim_0'))


def test_proper_unstack_dtype():
    """Test that we don't accidentally end up with dtype=O for the coords
    """
    a = xarray.DataArray(
        [[0, 1, 2, 3], [4, 5, 6, 7]],
        dims=['r', 'c'],
        coords={'r': pandas.to_datetime(['2000/01/01', '2000/01/02']),
                'c': [1, 2, 3, 4]})
    b = a.stack(s=['r', 'c'])
    c = proper_unstack(b, 's')
    xarray.testing.assert_equal(a, c)


def test_proper_unstack_mixed_coords():
    a = xarray.DataArray([[0, 1, 2, 3], [4, 5, 6, 7]],
                         dims=['r', 'c'],
                         coords={'r': [1, 'x0'],
                                 'c': [1, 2.2, '3', 'fourth']})
    b = a.stack(s=['r', 'c'])
    c = proper_unstack(b, 's')
    xarray.testing.assert_equal(a, c)


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
    xarray.testing.assert_equal(b, d)
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
