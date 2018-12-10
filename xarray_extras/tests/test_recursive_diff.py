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


def check(lhs, rhs, *expect, rel_tol=1e-09, abs_tol=0.0,
          brief_dims=()):
    expect = set(expect)
    actual = set(recursive_diff(
        lhs, rhs,
        rel_tol=rel_tol, abs_tol=abs_tol,
        brief_dims=brief_dims))
    assert actual == expect


@pytest.mark.parametrize('x', [
    123, 'blah', 'a\nb', math.nan, np.nan, True, False, [1, 2], (1, 2),
    np.int8(1), np.uint8(1), np.int64(1), np.uint64(1),
    np.float32(1), np.float64(1),
    {1: 2, 3: 4}, {1, 2}, frozenset([1, 2]),
    np.arange(10),
    np.arange(10, dtype=np.float64),
    pd.Series([1, 2]),
    pd.Series([1, 2], index=[3, 4]),
    pd.RangeIndex(10),
    pd.RangeIndex(1, 10, 3),
    pd.Index([1, 2, 3]),
    pd.MultiIndex.from_tuples([('bar', 'one'), ('bar', 'two'), ('baz', 'one')],
                              names=['l1', 'l2']),
    pd.DataFrame([[1, 2], [3, 4]]),
    pd.DataFrame([[1, 2], [3, 4]], index=['i1', 'i2'], columns=['c1', 'c2']),
    xarray.DataArray([1, 2]),
    xarray.DataArray([1, 2], dims=['x'], coords={'x': [3, 4]}),
    Rectangle(1, 2),
    Circle(1),
])
def test_identical(x):
    assert not list(recursive_diff(x, deepcopy(x)))


def test_simple():
    check(1, 0, '1 != 0 (abs: -1.0e+00, rel: -1.0e+00)')
    check('asd', 'lol', 'asd != lol')
    check(b'asd', b'lol', "b'asd' != b'lol'")
    check(True, False, 'True != False')


def test_object_type_differs():
    check(1, '1', '1 != 1', "object type differs: int != str")
    check(True, 1, "object type differs: bool != int")
    check(False, 0, "object type differs: bool != int")
    check([1, 2], (1, 2), "object type differs: list != tuple")
    check({1, 2}, frozenset([1, 2]), "object type differs: set != frozenset")


def test_collections():
    check([1, 2], [1, 2, 3], 'RHS has 1 more elements than LHS: [3]')
    check({1, 2}, {1, 2, (3, 4)}, "(3, 4) is in RHS only")
    check({'x': 10, 'y': 20}, {'x': 10, 'y': 30},
          '[y]: 20 != 30 (abs: 1.0e+01, rel: 5.0e-01)')
    check({2: 20}, {1: 10},
          "Pair 1:10 is in RHS only", "Pair 2:20 is in LHS only")


def test_limit_str_length():
    """Long and multi-line strings are truncated
    """
    check('a' * 100, 'a' * 100)
    check('a' * 100, 'a' * 101, '%s ... != %s ...' % ('a' * 76, 'a' * 76))
    check('a\nb', 'a\nb')
    check('a\nb', 'a\nc', 'a ... != a ...')


@pytest.mark.parametrize('nan', [np.nan, math.nan])
def test_nan(nan):
    check(nan, nan)
    check(nan, math.nan)
    check(nan, np.nan)
    check(0.0, nan, '0.0 != nan (abs: nan, rel: nan)')
    check(nan, 0.0, 'nan != 0.0 (abs: nan, rel: nan)')


def test_float():
    """Float comparison with tolerance
    """
    # Test that floats are not accidentally rounded when printing
    check(123456.7890123456, 123456.789,
          '123456.7890123456 != 123456.789 (abs: -1.2e-05, rel: -1.0e-10)',
          rel_tol=0, abs_tol=0)

    check(123, 123.0000000000001)  # difference is below rel_tol=1e-9

    check(123456.7890123456, 123456.789,
          '123456.7890123456 != 123456.789 (abs: -1.2e-05, rel: -1.0e-10)',
          rel_tol=1e-11, abs_tol=0)
    check(123456.7890123456, 123456.789,
          '123456.7890123456 != 123456.789 (abs: -1.2e-05, rel: -1.0e-10)',
          rel_tol=0, abs_tol=1e-5)

    check(123456.7890123456, 123456.789, rel_tol=0, abs_tol=1e-4)
    check(123456.7890123456, 123456.789, rel_tol=1e-7, abs_tol=0)

    # Abs tol is RHS - LHS; rel tol is RHS / LHS - 1
    check(80.0, 175.0, '80.0 != 175.0 (abs: 9.5e+01, rel: 1.2e+00)')

    # Division by zero in relative delta
    check(1.0, 0.0, '1.0 != 0.0 (abs: -1.0e+00, rel: -1.0e+00)')
    check(0.0, 1.0, '0.0 != 1.0 (abs: 1.0e+00, rel: nan)')

    # tolerance settings are retained when descending into containers
    check([{'x': (1.0, 2.0)}], [{'x': (1.1, 2.01)}],
          '[0][x][0]: 1.0 != 1.1 (abs: 1.0e-01, rel: 1.0e-01)',
          rel_tol=.05, abs_tol=0)

    # tolerance > 1 in a comparison among int's
    # note how int's are not cast to float when both lhs and rhs are int
    check(1, 2, abs_tol=2)
    check(2, 5, '2 != 5 (abs: 3.0e+00, rel: 1.5e+00)', abs_tol=2)


def test_int_vs_float():
    """ints are silently cast to float and do not cause an
    'object type differs' error.
    """
    check(123, 123.0)
    check(123, 123.0000000000001)  # difference is below rel_tol=1e-9
    check(1, 1.01, '1.0 != 1.01 (abs: 1.0e-02, rel: 1.0e-02)', abs_tol=.001)
    check(1, 1.01, abs_tol=.1)


def test_numpy_types():
    """scalar numpy data types (not to be confused with numpy arrays)
    are silently cast to pure numpy types and do not cause an
    'object type differs' error. They're compared with tolerance.
    """
    check(123, np.int32(123))
    check(np.int64(123), np.int32(123))
    check(123, np.float64(123))
    check(np.float32(123), np.float64(123))
    check(np.float64(1), np.float64(1.01),
          '1.0 != 1.01 (abs: 1.0e-02, rel: 1.0e-02)', abs_tol=.001)
    check(np.float32(1), np.float32(1.01), abs_tol=.1)
    check(np.float64(1), np.float64(1.01), abs_tol=.1)


def test_numpy():
    # test tolerance and comparison of float vs. int
    check(np.array([1.0, 2.0, 3.01, 4.0001, 5.0]),
          np.array([1, 4, 3, 4], dtype=np.int64),
          '[data][1]: 2.0 != 4.0 (abs: 2.0e+00, rel: 1.0e+00)',
          '[data][2]: 3.01 != 3.0 (abs: -1.0e-02, rel: -3.3e-03)',
          '[dim_0]: LHS has 1 more elements than RHS',
          'object type differs: ndarray<float64> != ndarray<int64>',
          abs_tol=0.001)

    # Tolerance > 1 in a comparison among int's
    # Make sure that tolerance is not applied to RangeIndex comparison
    check(np.array([1, 2]), np.array([2, 20, 3, 4]),
          '[data][1]: 2 != 20 (abs: 1.8e+01, rel: 9.0e+00)',
          '[dim_0]: RHS has 2 more elements than LHS',
          abs_tol=10)

    # array of numbers vs. dates; mismatched size
    check(np.array([1, 2], dtype=np.int64),
          pd.to_datetime(['2000-01-01', '2000-01-02', '2000-01-03']).values,
          '[data][0]: 1 != 2000-01-01 00:00:00',
          '[data][1]: 2 != 2000-01-02 00:00:00',
          '[dim_0]: RHS has 1 more elements than LHS',
          'object type differs: ndarray<int64> != ndarray<datetime64>')

    # array of numbers vs. strings; mismatched size
    check(np.array([1, 2, 3], dtype=np.int64),
          np.array(['foo', 'bar']),
          '[data][0]: 1 != foo',
          '[data][1]: 2 != bar',
          '[dim_0]: LHS has 1 more elements than RHS',
          'object type differs: ndarray<int64> != ndarray<<U...>')

    # Mismatched dimensions
    check(np.array([1, 2, 3, 4]),
          np.array([[1, 2], [3, 4]]),
          '[dim_0]: LHS has 2 more elements than RHS',
          'Dimension dim_1 is in RHS only')

    # numpy vs. list
    check(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
          [[1, 4, 3], [4, 5, 6]],
          "object type differs: ndarray<int64> != list",
          "[data][0, 1]: 2 != 4 (abs: 2.0e+00, rel: 1.0e+00)")

    # numpy vs. other object
    check(np.array([0, 0], dtype=np.int64), 0,
          "Dimension dim_0 is in LHS only",
          "object type differs: ndarray<int64> != int")


def test_numpy_strings():
    """Strings in numpy can be unicode (<U...), binary ascii (<S...)
    or Python variable-length (object).
    Test that these three types are not considered equivalent.
    """
    a = np.array(['foo'], dtype=object)
    b = np.array(['foo'], dtype='U')
    c = np.array(['foo'], dtype='S')
    check(a, b,
          "object type differs: ndarray<object> != ndarray<<U...>")
    check(a, c,
          "object type differs: ndarray<object> != ndarray<|S...>",
          "[data][0]: foo != b'foo'")
    check(b, c,
          "object type differs: ndarray<<U...> != ndarray<|S...>",
          "[data][0]: foo != b'foo'")


@pytest.mark.parametrize('x,y', [
    ('foo', 'barbaz'),
    (b'foo', b'babaz'),
])
def test_numpy_string_slice(x, y):
    """When slicing an array of strings, the output sub-dtype won't change.
    Test that string that differs only by dtype-length are considered
    equivalent.
    """
    a = np.array([x, y])  # dtype=<U6/<S6
    b = a[:1]  # dtype=<U6/<S6
    c = np.array([x])  # dtype=<U3/<S3
    assert a.dtype == b.dtype
    assert a.dtype != c.dtype
    check(b, c)


def test_numpy_dates():
    a = pd.to_datetime(
        ['2000-01-01',
         '2000-01-02',
         '2000-01-03',
         'NaT']).values.astype('<M8[D]')
    b = pd.to_datetime(
        ['2000-01-01',  # identical
         '2000-01-04',  # differs, both LHS and RHS are non-NaT
         'NaT',  # non-NaT vs. NaT
         'NaT',  # NaT == NaT
         # differences in sub-type must be ignored
         ]).values.astype('<M8[ns]')
    check(a, b,
          '[data][1]: 2000-01-02 00:00:00 != 2000-01-04 00:00:00',
          '[data][2]: 2000-01-03 00:00:00 != NaT')


def test_numpy_scalar():
    check(np.array(1, dtype=np.int64), np.array(2.5),
          '[data]: 1.0 != 2.5 (abs: 1.5e+00, rel: 1.5e+00)',
          'object type differs: ndarray<int64> != ndarray<float64>')
    check(np.array(1, dtype=np.int64), 2,
          '[data]: 1 != 2 (abs: 1.0e+00, rel: 1.0e+00)',
          'object type differs: ndarray<int64> != int')
    check(np.array('foo'), np.array('bar'),
          '[data]: foo != bar')
    # Note: datetime64 are not 0-dimensional arrays
    check(np.datetime64('2000-01-01'), np.datetime64('2000-01-02'),
          '2000-01-01 != 2000-01-02')
    check(np.datetime64('2000-01-01'), np.datetime64('NaT'),
          '2000-01-01 != NaT')


def test_pandas_series():
    # pd.Series
    # Note that we're also testing that order is ignored
    check(pd.Series([1, 2, 3], index=['foo', 'bar', 'baz'], name='hello'),
          pd.Series([1, 3, 4], index=['foo', 'baz', 'bar'], name='world'),
          "[data][index=bar]: 2 != 4 (abs: 2.0e+00, rel: 1.0e+00)",
          "[name]: hello != world")


def test_pandas_dataframe():
    df1 = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6]],
        index=['x1', 'x2'],
        columns=['y1', 'y2', 'y3'])
    df2 = pd.DataFrame(
        [[1, 3, 2], [4, 7, 5]],
        index=['x1', 'x2'],
        columns=['y1', 'y3', 'y4'])

    check(df1, df2,
          '[data][column=y3, index=x2]: 6 != 7 (abs: 1.0e+00, rel: 1.7e-01)',
          '[columns]: y2 is in LHS only',
          '[columns]: y4 is in RHS only')


def pandas_index():
    # Regular index
    # Test that order is ignored
    # Use huge abs_tol and rel_tol to test that tolerance is ignored
    check(pd.Index([1, 2, 3, 4]), pd.Index([1, 3.000001, 2]),
          '3.0 is in LHS only',
          '3.000001 is in RHS only',
          '4.0 is in LHS only',
          'object type differs: Int64Index != Float64Index',
          rel_tol=10, abs_tol=10)

    check(pd.Index(['x', 'y', 'z']), pd.Index(['y', 'x']),
          'z is in LHS only')


def test_pandas_rangeindex():
    # RangeIndex(stop)
    check(pd.RangeIndex(10), pd.RangeIndex(10))
    check(pd.RangeIndex(8), pd.RangeIndex(10),
          'RHS has 2 more elements than LHS')
    check(pd.RangeIndex(10), pd.RangeIndex(8),
          'LHS has 2 more elements than RHS')

    # RangeIndex(start, stop, step, name)
    check(pd.RangeIndex(1, 2, 3, name='x'),
          pd.RangeIndex(1, 2, 3, name='x'))
    check(pd.RangeIndex(0, 4, 1), pd.RangeIndex(1, 4, 1),
          'RangeIndex(start=0, stop=4, step=1) != '
          'RangeIndex(start=1, stop=4, step=1)')
    check(pd.RangeIndex(0, 4, 2), pd.RangeIndex(0, 5, 2),
          'RangeIndex(start=0, stop=4, step=2) != '
          'RangeIndex(start=0, stop=5, step=2)')
    check(pd.RangeIndex(0, 4, 2), pd.RangeIndex(0, 4, 3),
          'RangeIndex(start=0, stop=4, step=2) != '
          'RangeIndex(start=0, stop=4, step=3)')
    check(pd.RangeIndex(4, name='foo'), pd.RangeIndex(4, name='bar'),
          "RangeIndex(start=0, stop=4, step=1, name='foo') != "
          "RangeIndex(start=0, stop=4, step=1, name='bar')")

    # RangeIndex vs regular index
    check(pd.RangeIndex(4), pd.Index([0, 1, 2]),
          '3 is in LHS only',
          'object type differs: RangeIndex != Int64Index')


def test_pandas_multiindex():
    lhs = pd.MultiIndex.from_tuples(
        [('bar', 'one'), ('bar', 'two'), ('baz', 'one')],
        names=['l1', 'l2'])
    rhs = pd.MultiIndex.from_tuples(
        [('baz', 'one'), ('bar', 'three'), ('bar', 'one'), ('baz', 'four')],
        names=['l1', 'l3'])
    check(lhs, rhs,
          "[data]: ('bar', 'three') is in RHS only",
          "[data]: ('bar', 'two') is in LHS only",
          "[data]: ('baz', 'four') is in RHS only",
          '[names][1]: l2 != l3')

    # MultiIndex vs. regular index
    check(lhs, pd.Index([0, 1, 2]),
          "Cannot compare objects: MultiIndex(levels=[['bar', 'baz'], "
          "['one', 'two']], ..., Int64Index([0, 1, 2], dtype='int64')",
          'object type differs: MultiIndex != Int64Index')


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
        attrs={'some': 'attr', 'some2': 1})

    ds2 = ds1.copy(deep=True)
    del ds2['d1']
    ds2['d2'][0, 0] = 10
    ds2['nonindex'][1] = 'ni4'
    ds2.attrs['some2'] = 2
    ds2.attrs['other'] = 'someval'

    check(ds1, ds2,
          '[attrs]: Pair other:someval is in RHS only',
          '[attrs][some2]: 1 != 2 (abs: 1.0e+00, rel: 1.0e+00)',
          '[coords][nonindex][x=x2]: ni2 != ni4',
          "[data_vars]: Pair d1:<xarray.DataArray 'd1' (__stacked__: 3)> ... is in LHS only",  # noqa: E501
          '[data_vars][d2][x=x1, y=y1]: 4 != 10 (abs: 6.0e+00, rel: 1.5e+00)')

    check(ds1, ds2,
          '[attrs]: Pair other:someval is in RHS only',
          '[coords][nonindex][x=x2]: ni2 != ni4',
          "[data_vars]: Pair d1:<xarray.DataArray 'd1' (__stacked__: 3)> ... is in LHS only",  # noqa: E501
          abs_tol=7)

    # xarray.DataArray
    # Note: this sample has a non-index coordinate
    # In Linux, int maps to int64 while in Windows it maps to int32
    da1 = ds1['d2'].astype(np.int64)
    da1.name = 'foo'
    da1.attrs['attr1'] = 1.0
    da1.attrs['attr2'] = 1.0

    # Test dimension order does not matter
    check(da1, da1.T)

    da2 = da1.copy(deep=True).astype(float)
    da2[0, 0] *= 1.0 + 1e-7
    da2[0, 1] *= 1.0 + 1e-10
    da2['nonindex'][1] = 'ni4'
    da2.name = 'bar'
    da2.attrs['attr1'] = 1.0 + 1e-7
    da2.attrs['attr2'] = 1.0 + 1e-10
    da2.attrs['attr3'] = 'new'

    check(da1, da2,
          '[attrs]: Pair attr3:new is in RHS only',
          '[attrs][attr1]: 1.0 != 1.0000001 (abs: 1.0e-07, rel: 1.0e-07)',
          '[coords][nonindex][x=x2]: ni2 != ni4',
          '[data][x=x1, y=y1]: 4.0 != 4.0000004 (abs: 4.0e-07, rel: 1.0e-07)',
          '[name]: foo != bar',
          'object type differs: DataArray<int64> != DataArray<float64>')


def test_xarray_scalar():
    da1 = xarray.DataArray(1.0)
    da2 = xarray.DataArray(1.0 + 1e-7)
    check(da1, da2, '[data]: 1.0 != 1.0000001 (abs: 1.0e-07, rel: 1.0e-07)')
    da2 = xarray.DataArray(1.0 + 1e-10)
    check(da1, da2)


def test_xarray_no_coords():
    check(xarray.DataArray([0, 1]),
          xarray.DataArray([0, 2]),
          '[data][1]: 1 != 2 (abs: 1.0e+00, rel: 1.0e+00)')


def test_xarray_mismatched_dims():
    # 0-dimensional vs. 1+-dimensional
    check(xarray.DataArray(1.0),
          xarray.DataArray([0.0, 0.1]),
          "[index]: Dimension dim_0 is in RHS only")

    # both arrays are 1+-dimensional
    check(xarray.DataArray([0, 1], dims=['x']),
          xarray.DataArray([[0, 1], [2, 3]], dims=['x', 'y']),
          "[index]: Dimension y is in RHS only")


def test_xarray_size0():
    check(xarray.DataArray([]),
          xarray.DataArray([1.0]),
          '[index][dim_0]: RHS has 1 more elements than LHS')


def test_xarray_stacked():
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
    check(da1, da2,
          "[data][x=x1, y=0, z=0]: 0 != 10 (abs: 1.0e+01, rel: nan)",
          '[index]: Dimension s is in RHS only',
          '[index]: Dimension x is in LHS only',
          '[index]: Dimension y is in LHS only')


def test_brief_dims_1d():
    # all dims are brief
    da1 = xarray.DataArray([1, 2, 3], dims=['x'])
    da2 = xarray.DataArray([1, 3, 4], dims=['x'])
    check(da1, da2,
          '[data][x=1]: 2 != 3 (abs: 1.0e+00, rel: 5.0e-01)',
          '[data][x=2]: 3 != 4 (abs: 1.0e+00, rel: 3.3e-01)')
    check(da1, da2,
          '[data]: 2 differences',
          brief_dims=['x'])
    check(da1, da2,
          '[data]: 2 differences',
          brief_dims='all')

    check(da1, da1)
    check(da1, da1, brief_dims=['x'])
    check(da1, da1, brief_dims='all')


def test_brief_dims_nd():
    # some dims are brief
    da1 = xarray.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dims=['r', 'c'])
    da2 = xarray.DataArray([[1, 5, 4], [4, 5, 6], [7, 8, 0]], dims=['r', 'c'])
    check(da1, da2,
          '[data][c=1, r=0]: 2 != 5 (abs: 3.0e+00, rel: 1.5e+00)',
          '[data][c=2, r=0]: 3 != 4 (abs: 1.0e+00, rel: 3.3e-01)',
          '[data][c=2, r=2]: 9 != 0 (abs: -9.0e+00, rel: -1.0e+00)')
    check(da1, da2,
          '[data][c=1]: 1 differences',
          '[data][c=2]: 2 differences',
          brief_dims=['r'])
    check(da1, da2,
          '[data]: 3 differences',
          brief_dims='all')

    check(da1, da1)
    check(da1, da1, brief_dims=['r'])
    check(da1, da1, brief_dims='all')


def test_brief_dims_nested():
    """xarray object not at the first level, and not all variables have all
    brief_dims
    """
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
    check(lhs, rhs,
          '[foo][data_vars][x][c=2, r=0]: 3 != 4 (abs: 1.0e+00, rel: 3.3e-01)',
          '[foo][data_vars][y][c=2]: 3 != 4 (abs: 1.0e+00, rel: 3.3e-01)')
    check(lhs, rhs,
          '[foo][data_vars][x][c=2]: 1 differences',
          '[foo][data_vars][y][c=2]: 3 != 4 (abs: 1.0e+00, rel: 3.3e-01)',
          brief_dims=['r'])
    check(lhs, rhs,
          '[foo][data_vars][x]: 1 differences',
          '[foo][data_vars][y]: 1 differences',
          brief_dims='all')


def test_nested1():
    # Subclasses of the supported types must only produce a type error
    class MyDict(dict):
        pass

    class MyList(list):
        pass

    class MyTuple(tuple):
        pass

    # Two complex arrays which are identical
    lhs = {
        'foo': [1, 2, (5.2, 'asd')],
        'bar': None,
        'baz': np.array([1, 2, 3]),
        None: [np.array([1, 2, 3])]
    }
    rhs = MyDict({
        'foo': MyList([1, 2, MyTuple((5.20000000001, 'asd'))]),
        'bar': None,
        'baz': np.array([1, 2, 3]),
        None: [np.array([1, 2, 3])]
    })
    check(lhs, rhs,
          '[foo]: object type differs: list != MyList',
          '[foo][2]: object type differs: tuple != MyTuple',
          'object type differs: dict != MyDict')


def test_nested2():
    lhs = {
        'foo': [1, 2, ('asd', 5.2), 4],
        'bar': np.array([1, 2, 3, 4], dtype=np.int64),
        'baz': np.array([1, 2, 3], dtype=np.int64),
        'key_only_lhs': None,
    }
    rhs = {
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

    check(lhs, rhs,
          "[bar]: object type differs: ndarray<int64> != ndarray<float64>",
          "[bar][dim_0]: LHS has 1 more elements than RHS",
          "[baz]: object type differs: ndarray<int64> != list",
          "[foo]: LHS has 1 more elements than RHS: [4]",
          "[foo][2]: RHS has 1 more elements than LHS: [3]",
          "[foo][2]: object type differs: tuple != list",
          "[foo][2][0]: asd != lol",
          "Pair key_only_lhs:None is in LHS only",
          "Pair key_only_rhs:%s ... is in RHS only" % ('a' * 76))


def test_custom_classes():
    check(Rectangle(1, 2), Rectangle(1.1, 2.7),
          '[h]: 2.0 != 2.7 (abs: 7.0e-01, rel: 3.5e-01)',
          abs_tol=.5)

    check(Rectangle(1, 2), Drawing(3, 2),
          '[w]: 1 != 3 (abs: 2.0e+00, rel: 2.0e+00)',
          'object type differs: Rectangle != Drawing')

    # Unregistered classes can still be compared but without
    # tolerance or recursion
    check(Circle(4), Circle(4.1),
          'Circle(4.000000) != Circle(4.100000)',
          abs_tol=.5)

    check(Rectangle(4, 4), Square(4),
          'Cannot compare objects: Rectangle(4.000000, 4.000000), Square(4.000000)',  # noqa: E501
          'object type differs: Rectangle != Square')

    check(Circle(4), Square(4),
          'Cannot compare objects: Circle(4.000000), Square(4.000000)',
          'object type differs: Circle != Square')
