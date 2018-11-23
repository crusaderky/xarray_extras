from copy import deepcopy
import math
import numpy as np
import pandas as pd
import pytest
import xarray
from xarray_extras.recursive_diff import recursive_diff, cast


class Rectangle:
    """Sample class to test custom comparisons
    """
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __eq__(self, other):
        return self.w == other.w and self.h == other.h

    def __repr__(self):
        return 'Rectangle(%f, %f)' % (self.w, self.h)


class Drawing:
    """Another class that is not Rectangle but just happens to be cast to the
    same dict
    """
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __eq__(self, other):
        return self.w == other.w and self.h == other.h


@cast.register(Rectangle)
@cast.register(Drawing)
def _(obj, brief_dims):
    return {'w': obj.w, 'h': obj.h}


class Circle:
    """A class which that supports == but is not registered
    """
    def __init__(self, radius):
        self.radius = radius

    def __eq__(self, other):
        return self.radius == other.radius

    def __repr__(self):
        return 'Circle(%f)' % self.radius


class Square:
    """Another unregistered class
    """
    def __init__(self, side):
        self.side = side

    def __eq__(self, other):
        return self.side == other.side

    def __repr__(self):
        return 'Square(%f)' % self.side


@pytest.mark.parametrize('x', [
    123, 'blah', 'a\nb', math.nan, np.nan, True, False, [1, 2], (1, 2),
    {1: 2, 3: 4}, {1, 2}, frozenset([1, 2]),
    np.arange(10),
    np.arange(10, dtype=np.float64),
    pd.Series([1, 2]),
    pd.Series([1, 2], index=[3, 4]),
    pd.DataFrame([[1, 2], [3, 4]]),
    pd.DataFrame([[1, 2], [3, 4]], index=['i1', 'i2'], columns=['c1', 'c2']),
    xarray.DataArray([1, 2]),
    xarray.DataArray([1, 2], dims=['x'], coords={'x': [3, 4]}),
    Rectangle(1, 2),
    Circle(1),
])
def test_identical(x):
    assert not list(recursive_diff(x, deepcopy(x)))


@pytest.mark.parametrize('lhs,rhs', [
    (123, 123.0),
    (123, 123.0000000000001),  # difference is below rel_tol=1e-9
    (123.0, 123.0000000000001),
    (np.nan, math.nan),
    ({1: 10, 2: 20}, {1: 10, 2: 20.000000000001}),
])
def test_treat_as_identical(lhs, rhs):
    assert not list(recursive_diff(lhs, rhs))
    assert not list(recursive_diff(rhs, lhs))


#def test_recursive_diff(lhs, rhs, expect, rel_tol=1e-09, abs_tol=0.0, brief_dims=()):
#    expect = set(expect)
#    actual = set(recursive_diff(
#        lhs, rhs,
#        rel_tol=rel_tol, abs_tol=abs_tol,
#        brief_dims=brief_dims))
#    print("Expected:")
#    pprint.pprint(expect)
#    print("Got:")
#    pprint.pprint(actual)
#    eq_(expect, actual)


@pytest.mark.parametrize('lhs,rhs,diffs', [
    (1, 0, {'1 != 0 (abs: -1.0e+00, rel: -1.0e+00)'}),
    ('asd', 'lol', {'asd != lol'}),
    (1, '1', {'1 != 1', "object type differs: int != str"}),
    (True, 1, {"object type differs: bool != int"}),
    (False, 0, {"object type differs: bool != int"}),
    ([1, 2], (1, 2), {"object type differs: list != tuple"}),
    ([1, 2], [1, 2, 3], {'RHS has 1 more elements than LHS: [3]'}),
    ({1, 2}, frozenset([1, 2]), {"object type differs: set != frozenset"}),
    ({1, 2}, {1, 2, (3, 4)}, {"(3, 4) is in RHS only"}),
    ({'x': 10, 'y': 20}, {'x': 10, 'y': 30},
     {'[y]: 20 != 30 (abs: 1.0e+01, rel: 5.0e-01)'}),
    ({2: 20}, {1: 10},
     {"Pair 1:10 is in RHS only", "Pair 2:20 is in LHS only"}),
    # Long and multi-line strings are truncated
    ('a' * 100, 'a' * 101, {'%s ... != %s ...' % ('a' * 76, 'a' * 76)}),
    ('a\nb', 'a\nc', {'a ... != a ...'}),
])
def test_simple(lhs, rhs, diffs):
    actual = set(recursive_diff(lhs, rhs))
    assert diffs == actual


@pytest.mark.parametrize('lhs,rhs,rel_tol,abs_tol,diffs', [
    # Test that floats are not accidentally rounded when printing
    (123456.7890123456, 123456.789, 0, 0,
     {'123456.7890123456 != 123456.789 (abs: -1.2e-05, rel: -1.0e-10)'}),
    (123456.7890123456, 123456.789, 1e-11, 0,
     {'123456.7890123456 != 123456.789 (abs: -1.2e-05, rel: -1.0e-10)'}),
    (123456.7890123456, 123456.789, 0, 1e-5,
     {'123456.7890123456 != 123456.789 (abs: -1.2e-05, rel: -1.0e-10)']),
    (123456.7890123456, 123456.789, 0, 1e-4, set()),
    (123456.7890123456, 123456.789, 1e-7, 0, set()),

    # Abs tol is RHS - LHS; rel tol is RHS / LHS - 1
    (80.0, 175.0, 1e-9, 0, {'80.0 != 175.0 (abs: 9.5e+01, rel: 1.2e+00)'}),

    # Division by zero in relative delta
    (1.0, 0.0, 1e-9, 0, {'1.0 != 0.0 (abs: -1.0e+00, rel: -1.0e+00)'}),
    (0.0, 1.0, 1e-9, 0, {'0.0 != 1.0 (abs: 1.0e+00, rel: nan)'}),

    # lhs and/or rhs are NaN
    (0.0, math.nan, 1e-9, 0, {'0.0 != nan (abs: nan, rel: nan)'}),
    (math.nan, 0.0, 1e-9, 0, {'nan != 0.0 (abs: nan, rel: nan)'}),
    (0.0, np.nan, 1e-9, 0, {'0.0 != nan (abs: nan, rel: nan)'}),
    (np.nan, 0.0, 1e-9, 0, {'nan != 0.0 (abs: nan, rel: nan)'}),

    # tolerance settings are retained when descending into containers
    ([{'x': (1.0, 2.0)}], [{'x': (1.1, 2.01)}], .05, 0,
     {'[0][x][0]: 1.0 != 1.1 (abs: 1e-01, rel: 1e-01'}),

    # int vs. float
    (1, 1.01, 0, 1e-3, {'1.0 != 1.01 (abs: 1e-01, rel: 1e-01'}),
    (1, 1.01, 0, 1e-1, set()),

    # tolerance > 1 in a comparison among int's
    ([1, 2], [2, 5], 0, 2, {'[1]: 2 != 5 (abs: 3.0e+00, rel: 1.5e+00)'}),
])
def test_tolerance(lhs, rhs, rel_tol, abs_tol, diffs):
    """Float comparison with tolerance
    """
    actual = set(recursive_diff(lhs, rhs, rel_tol=rel_tol, abs_tol=abs_tol))
    assert diffs == actual


def test_numpy():
    # test tolerance and comparison of float vs. int
    yield check_recursive_diff, \
          np.array([1.0, 2.0, 3.01, 4.0001, 5.0]), \
          np.array([1, 4, 3, 4]), \
          ['[data][1]: 2.0 != 4.0 (abs: 2.0e+00, rel: 1.0e+00)',
           '[data][2]: 3.01 != 3.0 (abs: -1.0e-02, rel: -3.3e-03)',
           '[dim_0]: LHS has 1 more elements than RHS',
           'object type differs: ndarray<float64> != ndarray<int64>'], \
          0, 0.001

    # Tolerance > 1 in a comparison among int's
    # Make sure that tolerance is not applied to RangeIndex comparison
    yield check_recursive_diff, \
          np.array([1, 2]), np.array([2, 20, 3, 4]), \
          ['[data][1]: 2 != 20 (abs: 1.8e+01, rel: 9.0e+00)',
           '[dim_0]: RHS has 2 more elements than LHS'], \
          0, 10

    # array of numbers vs. dates; mismatched size
    yield check_recursive_diff, \
          np.array([1, 2]), \
          pd.to_datetime(['2000-01-01', '2000-01-02', '2000-01-03']).values, \
          ['[data][0]: 1 != 2000-01-01 00:00:00',
           '[data][1]: 2 != 2000-01-02 00:00:00',
           '[dim_0]: RHS has 1 more elements than LHS',
           'object type differs: ndarray<int64> != ndarray<datetime64>']

    # array of numbers vs. strings; mismatched size
    yield check_recursive_diff, \
          np.array([1, 2, 3]), \
          np.array(['foo', 'bar']), \
          ['[data][0]: 1 != foo',
           '[data][1]: 2 != bar',
           '[dim_0]: LHS has 1 more elements than RHS',
           'object type differs: ndarray<int64> != ndarray<<U...>']

    # Mismatched dimensions
    yield check_recursive_diff, \
        np.array([1, 2, 3, 4]), \
        np.array([[1, 2], [3, 4]]), \
        ['[dim_0]: LHS has 2 more elements than RHS',
         'Dimension dim_1 is in RHS only']

    # numpy vs. list
    yield check_recursive_diff, \
          np.array([[1, 2, 3], [4, 5, 6]]), \
          [[1, 4, 3], [4, 5, 6]], \
          ["object type differs: ndarray<int64> != list",
           "[data][0, 1]: 2 != 4 (abs: 2.0e+00, rel: 1.0e+00)"]

    # numpy vs. other object
    yield check_recursive_diff, \
          np.array([0, 0]), 0, \
          ["Dimension dim_0 is in LHS only",
           "object type differs: ndarray<int64> != int"]


def test_numpy_strings():
    # Strings in numpy can be unicode (<U...), binary ascii (<S...)
    # or Python variable-length (object).
    # Test that these three types are not considered equivalent.
    a = np.array(['foo'], dtype=object)
    b = np.array(['foo'], dtype='U')
    c = np.array(['foo'], dtype='S')
    yield check_recursive_diff, a, b, ["object type differs: ndarray<object> != ndarray<<U...>"]
    yield check_recursive_diff, a, c, ["object type differs: ndarray<object> != ndarray<|S...>",
                                       "[data][0]: foo != b'foo'"]
    yield check_recursive_diff, b, c, ["object type differs: ndarray<<U...> != ndarray<|S...>",
                                       "[data][0]: foo != b'foo'"]

    # When slicing an array of strings, the output sub-dtype won't change.
    # Test that string that differs only by dtype-length are considered equivalent.
    a = np.array(['foo', 'barbaz'])  # dtype='<U6'
    b = a[:1]  # dtype='<U6'
    c = np.array(['foo'])  #dtype='<U3'
    assert a.dtype == b.dtype
    assert a.dtype != c.dtype
    yield check_recursive_diff, b, c, []

    a_s = a.astype('S')
    b_s = a_s[:1]
    c_s = c.astype('S')
    yield check_recursive_diff, b_s, c_s, []


def test_numpy_dates():
    a = pd.to_datetime(
        ['2000-01-01',
         '2000-01-02',
         '2000-01-03',
         'NaT'
        ]).values.astype('<M8[D]')
    b = pd.to_datetime([
        '2000-01-01',  # identical
        '2000-01-04',  # differs, both LHS and RHS are non-NaT
        'NaT',  # non-NaT vs. NaT
        'NaT',  # NaT == NaT
        # differences in sub-type must be ignored
        ]).values.astype('<M8[ns]')

    yield check_recursive_diff, a, b, [
        '[data][1]: 2000-01-02 00:00:00 != 2000-01-04 00:00:00',
        '[data][2]: 2000-01-03 00:00:00 != NaT',
    ]


def test_numpy_scalar():
    yield check_recursive_diff, \
        np.array(1), np.array(2.5), [
            '[data]: 1.0 != 2.5 (abs: 1.5e+00, rel: 1.5e+00)',
            'object type differs: ndarray<int64> != ndarray<float64>'
        ]

    yield check_recursive_diff, \
        np.array(1), 2, [
            '[data]: 1 != 2 (abs: 1.0e+00, rel: 1.0e+00)',
            'object type differs: ndarray<int64> != int',
        ]

    yield check_recursive_diff, \
        np.array('foo'), np.array('bar'), [
            '[data]: foo != bar'
        ]

    # Note: datetime64 are not 0-dimensional arrays
    yield check_recursive_diff, \
        np.datetime64('2000-01-01'), np.datetime64('2000-01-02'), [
            '2000-01-01 != 2000-01-02'
        ]

    yield check_recursive_diff, \
        np.datetime64('2000-01-01'), np.datetime64('NaT'), [
            '2000-01-01 != NaT'
        ]


def test_pandas():
    # pd.Series
    # Note that we're also testing that order is ignored
    yield check_recursive_diff, \
        pd.Series([1, 2, 3], index=['foo', 'bar', 'baz'], name='hello'), \
        pd.Series([1, 3, 4], index=['foo', 'baz', 'bar'], name='world'), \
        ["[data][index=bar]: 2 != 4 (abs: 2.0e+00, rel: 1.0e+00)",
         "[name]: hello != world"]

    # pd.DataFrame
    df1 = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6]],
        index=['x1', 'x2'],
        columns=['y1', 'y2', 'y3'])
    df2 = pd.DataFrame(
        [[1, 3, 2], [4, 7, 5]],
        index=['x1', 'x2'],
        columns=['y1', 'y3', 'y4'])

    yield check_recursive_diff, \
        df1, df2, \
        ['[data][column=y3, index=x2]: 6 != 7 (abs: 1.0e+00, rel: 1.7e-01)',
         '[columns]: y2 is in LHS only',
         '[columns]: y4 is in RHS only']


def test_index():
    # RangeIndex(stop)
    yield check_recursive_diff, \
        pd.RangeIndex(10), pd.RangeIndex(10), []
    yield check_recursive_diff, \
        pd.RangeIndex(8), pd.RangeIndex(10), \
        ['RHS has 2 more elements than LHS']
    yield check_recursive_diff, \
        pd.RangeIndex(10), pd.RangeIndex(8), \
        ['LHS has 2 more elements than RHS']

    # RangeIndex(start, stop, step, name)
    yield check_recursive_diff, \
        pd.RangeIndex(1, 2, 3, name='x'), \
        pd.RangeIndex(1, 2, 3, name='x'), []
    yield check_recursive_diff, \
        pd.RangeIndex(0, 4, 1), pd.RangeIndex(1, 4, 1), \
        ['RangeIndex(start=0, stop=4, step=1) != '
         'RangeIndex(start=1, stop=4, step=1)']
    yield check_recursive_diff, \
        pd.RangeIndex(0, 4, 2), pd.RangeIndex(0, 5, 2), \
        ['RangeIndex(start=0, stop=4, step=2) != '
         'RangeIndex(start=0, stop=5, step=2)']
    yield check_recursive_diff, \
        pd.RangeIndex(0, 4, 2), pd.RangeIndex(0, 4, 3), \
        ['RangeIndex(start=0, stop=4, step=2) != '
         'RangeIndex(start=0, stop=4, step=3)']
    yield check_recursive_diff, \
        pd.RangeIndex(4, name='foo'), pd.RangeIndex(4, name='bar'), \
        ["RangeIndex(start=0, stop=4, step=1, name='foo') != "
         "RangeIndex(start=0, stop=4, step=1, name='bar')"]
   
    # Regular index
    # Tolerance and order are ignored
    yield check_recursive_diff, \
        pd.Index([1, 2, 3, 4]), \
        pd.Index([1, 3.000001, 2]), [
            '3.0 is in LHS only',
            '3.000001 is in RHS only',
            '4.0 is in LHS only',
            'object type differs: Int64Index != Float64Index'],  \
            10, 10  # Huge abs_tol and rel_tol

    yield check_recursive_diff, \
        pd.Index(['x', 'y', 'z']), \
        pd.Index(['y', 'x']), \
        ['z is in LHS only']

    # MultiIndex
    lhs = pd.MultiIndex.from_tuples(
        [('bar', 'one'), ('bar', 'two'), ('baz', 'one')],
        names=['l1', 'l2'])
    rhs = pd.MultiIndex.from_tuples(
        [('baz', 'one'), ('bar', 'three'), ('bar', 'one'), ('baz', 'four')],
        names=['l1', 'l3'])
    yield check_recursive_diff, lhs, rhs, [
        "[data]: ('bar', 'three') is in RHS only",
         "[data]: ('bar', 'two') is in LHS only",
         "[data]: ('baz', 'four') is in RHS only",
         '[names][1]: l2 != l3']

    # MultiIndex vs. regular index
    yield check_recursive_diff, \
        lhs, pd.Index([0, 1, 2]), \
        ["Cannot compare objects: MultiIndex(levels=[['bar', 'baz'], ['one', 'two']], "
         "..., Int64Index([0, 1, 2], dtype='int64')",
         'object type differs: MultiIndex != Int64Index']

    # RangeIndex vs regular index
    yield check_recursive_diff, \
        pd.RangeIndex(4), pd.Index([0, 1, 2]), \
        ['3 is in LHS only',
         'object type differs: RangeIndex != Int64Index']


def test_xarray():
    # xarray.Dataset
    ds1 = xarray.Dataset(
        data_vars={
            'd1': ('x', [1, 2, 3]),
            'd2': (('y', 'x'), [[4, 5, 6], [7, 8, 9]]),
        },
        coords={
            'x': ('x', ['x1', 'x2', 'x3']),
            'y': ('y', ['y1', 'y2']),
            'nonindex': ('x', ['ni1', 'ni2', 'ni3']),
        },
        attrs={'some': 'attr'})

    ds2 = ds1.copy(deep=True)
    del ds2['d1']
    ds2['d2'][0, 0] = 10
    ds2['nonindex'][1] = 'ni4'
    ds2.attrs['other'] = 'someval'

    yield check_recursive_diff, \
        ds1, ds2, \
        ['[attrs]: Pair other:someval is in RHS only',
         '[coords][nonindex][x=x2]: ni2 != ni4',
         "[data_vars]: Pair d1:<xarray.DataArray 'd1' (__stacked__: 3)> ... is in LHS only",
         '[data_vars][d2][x=x1, y=y1]: 4 != 10 (abs: 6.0e+00, rel: 1.5e+00)']

    # xarray.DataArray
    # Note: this sample has a non-index coordinate
    da1 = ds1['d2']
    da1.name = 'foo'
    da1.attrs['attr1'] = 1.0
    da1.attrs['attr2'] = 1.0
    # Test dimension order does not matter
    yield check_recursive_diff, da1, da1.T, []

    da2 = da1.copy(deep=True).astype(float)
    da2[0, 0] *= 1.0 + 1e-7
    da2[0, 1] *= 1.0 + 1e-10
    da2['nonindex'][1] = 'ni4'
    da2.name = 'bar'
    da2.attrs['attr1'] = 1.0 + 1e-7
    da2.attrs['attr2'] = 1.0 + 1e-10
    da2.attrs['attr3'] = 'new'

    yield check_recursive_diff, da1, da2, \
        ['[attrs]: Pair attr3:new is in RHS only',
         '[attrs][attr1]: 1.0 != 1.0000001 (abs: 1.0e-07, rel: 1.0e-07)',
         '[coords][nonindex][x=x2]: ni2 != ni4',
         '[data][x=x1, y=y1]: 4.0 != 4.0000004 (abs: 4.0e-07, rel: 1.0e-07)',
         '[name]: foo != bar',
         'object type differs: DataArray<int64> != DataArray<float64>']

    # 0-dimensional inputs
    da1 = xarray.DataArray(1.0)
    da2 = xarray.DataArray(1.0 + 1e-7)
    yield check_recursive_diff, da1, da2, ['[data]: 1.0 != 1.0000001 (abs: 1.0e-07, rel: 1.0e-07)']
    da2 = xarray.DataArray(1.0 + 1e-10)
    yield check_recursive_diff, da1, da2, []

    # 1-dimensional input
    da1 = xarray.DataArray([0, 1])
    da2 = xarray.DataArray([0, 2])
    yield check_recursive_diff, da1, da2, ['[data][1]: 1 != 2 (abs: 1.0e+00, rel: 1.0e+00)']

    # Mismatched dims: 0-dimensional vs. 1+-dimensional
    da1 = xarray.DataArray(1.0)
    da2 = xarray.DataArray([0.0, 0.1])
    yield check_recursive_diff, da1, da2, ["[index]: Dimension dim_0 is in RHS only"]

    # Mismatched dims: both arrays are 1+-dimensional
    da1 = xarray.DataArray([0, 1], dims=['x'])
    da2 = xarray.DataArray([[0, 1], [2, 3]], dims=['x', 'y'])
    yield check_recursive_diff, da1, da2, ["[index]: Dimension y is in RHS only"]

    # Pre-stacked dims, mixed with non-stacked ones
    da1 = xarray.DataArray(
        [[[0, 1], [2, 3]],
         [[4, 5], [6, 7]]],
        dims=['x', 'y', 'z'],
        coords={'x': ['x1', 'x2']})

    # Stacked and unstacked dims are compared point by point,
    # while still pointing out the difference in stacking
    da2 = da1.copy(deep=True)
    da2[0, 0, 0] = 10
    da2 = da2.stack(s=['x', 'y'])
    yield check_recursive_diff, da1, da2, \
        ["[data][x=x1, y=0, z=0]: 0 != 10 (abs: 1.0e+01, rel: nan)",
         '[index]: Dimension s is in RHS only',
         '[index]: Dimension x is in LHS only',
         '[index]: Dimension y is in LHS only']

    # 0-elements array
    da1 = xarray.DataArray([])
    da2 = xarray.DataArray([1.0])
    yield check_recursive_diff, da1, da2, \
        ['[index][dim_0]: RHS has 1 more elements than LHS']


def test_brief_dims():
    # all dims are brief
    da1 = xarray.DataArray([1, 2, 3], dims=['x'])
    da2 = xarray.DataArray([1, 3, 4], dims=['x'])
    yield check_recursive_diff, da1, da2, [
        '[data][x=1]: 2 != 3 (abs: 1.0e+00, rel: 5.0e-01)',
        '[data][x=2]: 3 != 4 (abs: 1.0e+00, rel: 3.3e-01)']
    yield check_recursive_diff, da1, da2, \
        ['[data]: 2 differences'], 0, 0, ['x']
    yield check_recursive_diff, da1, da2, \
        ['[data]: 2 differences'], 0, 0, 'all'
    yield check_recursive_diff, da1, da1, [], 0, 0, ['x']

    # some dims are brief
    da1 = xarray.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dims=['r', 'c'])
    da2 = xarray.DataArray([[1, 5, 4], [4, 5, 6], [7, 8, 0]], dims=['r', 'c'])
    yield check_recursive_diff, da1, da2, [
        '[data][c=1, r=0]: 2 != 5 (abs: 3.0e+00, rel: 1.5e+00)',
        '[data][c=2, r=0]: 3 != 4 (abs: 1.0e+00, rel: 3.3e-01)',
        '[data][c=2, r=2]: 9 != 0 (abs: -9.0e+00, rel: -1.0e+00)']
    yield check_recursive_diff, da1, da2, [
        '[data][c=1]: 1 differences',
        '[data][c=2]: 2 differences'], 0, 0, ['r']
    yield check_recursive_diff, da1, da2, \
        ['[data]: 3 differences'], 0, 0, 'all'
    yield check_recursive_diff, da1, da1, [], 0, 0, ['r']

    # xarray object not at the first level, and not all
    # variables have all brief_dims
    lhs = {
        'foo': xarray.Dataset(
            data_vars={
                'x': (('r', 'c'), [[1, 2, 3], [4, 5, 6]]),
                'y': ('c', [1, 2, 3]),
            })
    }
    rhs = {
        'foo': xarray.Dataset(
            data_vars={
                'x': (('r', 'c'), [[1, 2, 4], [4, 5, 6]]),
                'y': ('c', [1, 2, 4]),
            })
    }
    yield check_recursive_diff, lhs, rhs, [
        '[foo][data_vars][x][c=2, r=0]: 3 != 4 (abs: 1.0e+00, rel: 3.3e-01)',
        '[foo][data_vars][y][c=2]: 3 != 4 (abs: 1.0e+00, rel: 3.3e-01)']
    yield check_recursive_diff, lhs, rhs, [
        '[foo][data_vars][x][c=2]: 1 differences',
        '[foo][data_vars][y][c=2]: 3 != 4 (abs: 1.0e+00, rel: 3.3e-01)'], 0, 0, ['r']
    yield check_recursive_diff, lhs, rhs, [
        '[foo][data_vars][x]: 1 differences',
        '[foo][data_vars][y]: 1 differences'], 0, 0, 'all'


def test_complex1():
    # Subclasses of the supported types must only produce a type error
    class MyDict(dict):
        pass
    class MyList(list):
        pass
    class MyTuple(tuple):
        pass

    # Two complex arrays which are identical
    LHS = {
        'foo': [1, 2, (5.2, 'asd')],
        'bar': None,
        'baz': np.array([1, 2, 3]),
        None: [np.array([1, 2, 3])]
    }
    RHS = MyDict({
        'foo': MyList([1, 2, MyTuple((5.20000000001, 'asd'))]),
        'bar': None,
        'baz': np.array([1, 2, 3]),
        None: [np.array([1, 2, 3])]
    })
    yield check_recursive_diff, LHS, RHS, [
        '[foo]: object type differs: list != MyList',
        '[foo][2]: object type differs: tuple != MyTuple',
        'object type differs: dict != MyDict',
    ]


def test_complex2():
    LHS = {
        'foo': [1, 2, ('asd', 5.2), 4],
        'bar': np.array([1, 2, 3, 4]),
        'baz': np.array([1, 2, 3]),
        'key_only_lhs': None,
    }
    RHS = {
        # type changed from tuple to list
        # a string content has changed
        # LHS outermost list is longer
        # RHS innermost list is longer
        'foo': [1, 2, ['lol', 5.2, 3]],
        # numpy dtype has changed
        # LHS is longer
        'bar': np.array([1, 2, 3], dtype=np.float64),
        # numpy vs. list
        'baz': [1, 2, 3],
        # Test string truncation
        'key_only_rhs': 'a' * 200,
    }

    yield check_recursive_diff, LHS, RHS, [
        "[bar]: object type differs: ndarray<int64> != ndarray<float64>",
        "[bar][dim_0]: LHS has 1 more elements than RHS",
        "[baz]: object type differs: ndarray<int64> != list",
        "[foo]: LHS has 1 more elements than RHS: [4]",
        "[foo][2]: RHS has 1 more elements than LHS: [3]",
        "[foo][2]: object type differs: tuple != list",
        "[foo][2][0]: asd != lol",
        "Pair key_only_lhs:None is in LHS only",
        "Pair key_only_rhs:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ... is in RHS only"
    ]



def test_custom_classes():
    yield check_recursive_diff, \
        Rectangle(1, 2), Rectangle(1.1, 2.7), \
        ['[h]: 2.0 != 2.7 (abs: 7.0e-01, rel: 3.5e-01)'], \
        0, .5
 
    yield check_recursive_diff, \
        Rectangle(1, 2), Drawing(3, 2), \
        ['[w]: 1 != 3 (abs: 2.0e+00, rel: 2.0e+00)',
         'object type differs: Rectangle != Drawing']
 
    yield check_recursive_diff, \
        Circle(4), Circle(4), []

    # Unregistered classes can still be compared but without
    # tolerance or recursion
    yield check_recursive_diff, \
        Circle(4), Circle(4.1), \
        ['Circle(4.000000) != Circle(4.100000)'], \
        0, .5

    yield check_recursive_diff, \
        Rectangle(4, 4), Square(4), \
        ['Cannot compare objects: Rectangle(4.000000, 4.000000), Square(4.000000)',
         'object type differs: Rectangle != Square']

    yield check_recursive_diff, \
        Circle(4), Square(4), \
        ['Cannot compare objects: Circle(4.000000), Square(4.000000)',
         'object type differs: Circle != Square']
