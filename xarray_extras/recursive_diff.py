"""Recursively compare Python objects.

See also its most commonly used wrapper:
:func:`~xarray_extras.testing.recursive_eq`
"""
import math
import re
from functools import singledispatch

import dask
import numpy
import pandas
import xarray

from .stack import proper_unstack


__all__ = ('recursive_diff', 'cast')


def recursive_diff(lhs, rhs, *, rel_tol=1e-09, abs_tol=0.0, brief_dims=()):
    """Compare two objects and yield all differences.
    The two objects must any of:

    - basic types (str, int, float, bool)
    - basic collections (list, tuple, dict, set, frozenset)
    - :class:`numpy.ndarray`
    - :class:`pandas.Series`
    - :class:`pandas.DataFrame`
    - :class:`pandas.Index`
    - :class:`xarray.DataArray`
    - :class:`xarray.Dataset`
    - any recursive combination of the above
    - any other object (compared with ==)

    Special treatment is reserved to different types:

    - floats and ints are compared with tolerance, using :func:`math.isclose`
    - NaN equals to NaN
    - bools are only equal to other bools
    - numpy arrays are compared elementwise and with tolerance,
      also testing the dtype
    - pandas and xarray objects are compared elementwise, with tolerance, and
      without order, and do not support duplicate indexes
    - xarray dimensions and variables are compared without order
    - collections (list, tuple, dict, set, frozenset) are recursively
      descended into
    - generic/unknown objects are compared with ==

    Custom classes can be registered to benefit from the above behaviour;
    see documentation in :func:`cast`.

    :param lhs:
        left-hand-side data structure
    :param rhs:
        right-hand-side data structure
    :param float rel_tol:
        relative tolerance when comparing numbers.
        Applies to floats, integers, and all numpy-based data.
    :param float abs_tol:
        absolute tolerance when comparing numbers.
        Applies to floats, integers, and all numpy-based data.
    :param brief_dims:
        One of:

        - sequence of strings representing xarray dimensions. If one or more
          differences are found along one of these dimensions, only one message
          will be reported, stating the differences count.
        - "all", to produce one line only for every xarray variable that
          differs

        Omit to output a line for every single different cell.

    Yields strings containing difference messages, prepended by the path to
    the point that differs.
    """
    yield from _recursive_diff(
        lhs, rhs, rel_tol=rel_tol, abs_tol=abs_tol, brief_dims=brief_dims,
        path=[], suppress_type_diffs=False, join='inner')


def _recursive_diff(lhs, rhs, *, rel_tol, abs_tol, brief_dims, path,
                    suppress_type_diffs, join):
    """Recursive implementation of :func:`recursive_diff`

    :param list path:
        list of nodes traversed so far, to be prepended to all error messages
    :param bool suppress_type_diffs:
        if True, don't print out messages about differeces in type
    :param str join:
        join type of numpy objects: 'inner' or 'outer'.
        Ignored for plain Python collections (set, dict, etc.) for which
        outer join is always applied.

    This function calls itself recursively for all elements of numpy-based
    data, list, tuple, and dict.values(). Every time, it appends to the
    path list one element.
    """
    def diff(msg, print_path=path):
        """Format diff message, prepending the formatted path
        """
        path_prefix = "".join("[%s]" % elem for elem in print_path)
        if path_prefix != '':
            path_prefix += ': '
        return path_prefix + msg

    def is_array(dtype):
        return any(
            dtype.startswith(t)
            for t in ('ndarray', 'DataArray', 'Series', 'DataFrame'))

    def is_array_like(dtype):
        return dtype in {'int', 'float', 'complex', 'bool',
                         'str', 'list', 'tuple'}

    def are_instances(lhs, rhs, cls):
        return isinstance(lhs, cls) and isinstance(rhs, cls)

    # Build string representation of the two variables *before* casting
    lhs_repr = _str_trunc(lhs)
    rhs_repr = _str_trunc(rhs)

    # Identify if the variables are indices that must go through outer join,
    # *before* casting. This will be propagated downwards into the recursion.
    if join == 'inner' and are_instances(lhs, rhs, pandas.Index):
        join = 'outer'

    if (are_instances(lhs, rhs, xarray.DataArray)
            and '__strip_dataarray__' in lhs.attrs
            and '__strip_dataarray__' in rhs.attrs):
        # Don't repeat dtype comparisons
        suppress_type_diffs = True

    # cast lhs and rhs to simpler data types; pretty-print data type
    dtype_lhs = _dtype_str(lhs)
    dtype_rhs = _dtype_str(rhs)
    lhs = cast(lhs, brief_dims=brief_dims)
    rhs = cast(rhs, brief_dims=brief_dims)

    # 1.0 vs. 1 must not be treated as a difference
    if isinstance(lhs, int) and isinstance(rhs, float):
        # Cast lhs to float
        dtype_lhs = 'float'
        lhs = float(lhs)
    elif isinstance(rhs, int) and isinstance(lhs, float):
        # Cast rhs to float
        dtype_rhs = 'float'
        rhs = float(rhs)

    # When comparing an array vs. a plain python list or scalar, log an error
    # for the different dtype and then proceed to compare the contents
    if is_array(dtype_lhs) and is_array_like(dtype_rhs):
        rhs = cast(numpy.array(rhs), brief_dims=brief_dims)
    elif is_array(dtype_rhs) and is_array_like(dtype_lhs):
        lhs = cast(numpy.array(lhs), brief_dims=brief_dims)

    # Allow mismatched comparison of a RangeIndex vs. a regular index
    if (isinstance(lhs, pandas.RangeIndex) and not
            isinstance(rhs, pandas.RangeIndex)):
        lhs = cast(pandas.Index(lhs.values), brief_dims=brief_dims)
    if (isinstance(rhs, pandas.RangeIndex) and not
            isinstance(lhs, pandas.RangeIndex)):
        rhs = cast(pandas.Index(rhs.values), brief_dims=brief_dims)

    if dtype_lhs != dtype_rhs and not suppress_type_diffs:
        yield diff("object type differs: %s != %s" % (dtype_lhs, dtype_rhs))

    # Continue even in case dtype doesn't match
    # This allows comparing e.g. a numpy array vs. a list or a tuple

    if are_instances(lhs, rhs, list):
        if len(lhs) > len(rhs):
            yield diff("LHS has %d more elements than RHS: %s" %
                       (len(lhs) - len(rhs), _str_trunc(lhs[len(rhs):])))
        elif len(lhs) < len(rhs):
            yield diff("RHS has %d more elements than LHS: %s" %
                       (len(rhs) - len(lhs), _str_trunc(rhs[len(lhs):])))
        for i, (lhs_i, rhs_i) in enumerate(zip(lhs, rhs)):
            yield from _recursive_diff(
                lhs_i, rhs_i, rel_tol=rel_tol, abs_tol=abs_tol,
                brief_dims=brief_dims, path=path + [i],
                suppress_type_diffs=suppress_type_diffs, join=join)

    elif are_instances(lhs, rhs, set):
        for x in sorted(lhs - rhs, key=repr):
            yield diff("%s is in LHS only" % _str_trunc(x))
        for x in sorted(rhs - lhs, key=repr):
            yield diff("%s is in RHS only" % _str_trunc(x))

    elif are_instances(lhs, rhs, pandas.RangeIndex):
        # Pretty-print differences in size. This is used not only by
        # pandas.Series and pandas.DataFrame, but also by numpy arrays
        # and xarrays without coords
        if (lhs._start == rhs._start == 0
                and lhs._step == rhs._step == 1
                and lhs.name == rhs.name):
            delta = rhs._stop - lhs._stop
            if delta < 0:
                yield diff("LHS has %d more elements than RHS" % -delta)
            elif delta > 0:
                yield diff("RHS has %d more elements than LHS" % delta)
        else:
            # General case
            # e.g. RangeIndex(start=1, stop=3, step=1, name='x')
            lhs, rhs = str(lhs), str(rhs)
            if lhs != rhs:
                yield diff("%s != %s" % (lhs, rhs))

    elif are_instances(lhs, rhs, dict):
        for key in sorted(lhs.keys() - rhs.keys(), key=repr):
            if isinstance(lhs[key], pandas.Index):
                join = 'outer'
            if join == 'outer':
                # Comparing an index
                yield diff("Dimension %s is in LHS only" % key)
            else:
                yield diff("Pair %s:%s is in LHS only" % (
                    key, _str_trunc(lhs[key])))
        for key in sorted(rhs.keys() - lhs.keys(), key=repr):
            if isinstance(rhs[key], pandas.Index):
                join = 'outer'
            if join == 'outer':
                # Comparing an index
                yield diff("Dimension %s is in RHS only" % key)
            else:
                yield diff("Pair %s:%s is in RHS only" % (
                    key, _str_trunc(rhs[key])))
        for key in sorted(rhs.keys() & lhs.keys(), key=repr):
            yield from _recursive_diff(
                lhs[key], rhs[key], rel_tol=rel_tol, abs_tol=abs_tol,
                brief_dims=brief_dims, path=path + [key],
                suppress_type_diffs=suppress_type_diffs, join=join)

    elif are_instances(lhs, rhs, bool):
        if lhs != rhs:
            yield diff('%s != %s' % (lhs, rhs))
    elif are_instances(lhs, rhs, str):
        if lhs != rhs:
            yield diff('%s != %s' % (lhs_repr, rhs_repr))
    elif are_instances(lhs, rhs, bytes):
        if lhs != rhs:
            yield diff('%s != %s' % (lhs_repr, rhs_repr))
    elif are_instances(lhs, rhs, (int, float, complex)):
        if math.isnan(lhs) and math.isnan(rhs):
            pass
        elif not math.isclose(lhs, rhs, rel_tol=rel_tol, abs_tol=abs_tol):
            try:
                rel_delta = rhs / lhs - 1
            except ZeroDivisionError:
                rel_delta = math.nan
            yield diff('%s != %s (abs: %.1e, rel: %.1e)' %
                       (lhs, rhs, rhs - lhs, rel_delta))

    elif are_instances(lhs, rhs, xarray.DataArray):
        # This block is executed for all data that was originally:
        # - numpy.ndarray
        # - pandas.Series
        # - pandas.DataFrame
        # - pandas.Index (except RangeIndex)
        # - xarray.DataArray
        # - xarray.Dataset
        # - any of the above, compared against a plain Python list

        # Both DataArrays are guaranteed by _strip_dataarray to be either
        # ravelled on a single dim with a MultiIndex or 0-dimensional

        lhs_dims = _get_stripped_dims(lhs)
        rhs_dims = _get_stripped_dims(rhs)

        if lhs_dims != rhs_dims:
            # This is already reported elsewhere when comparing dicts
            # (Dimension x is in LHS only)
            pass

        elif lhs.dims:
            # Load the entire objects into RAM. When parsing huge disk-backed
            # datasets, e.g. with landg.bin.ncdiff, you want to do this at the
            # very last possible moment. After this, we'll do:
            # - alignment, which is potentially very expensive with dask
            # - Extract differences (simplified code):
            #     diff = lhs != rhs
            #     lhs = lhs[lhs != rhs].compute()
            #     rhs = rhs[lhs != rhs].compute()
            #   The above 3 lines, if lhs and rhs were dask-backed, would
            #   effectively load the arrays 3 times each.
            lhs, rhs = dask.compute(lhs, rhs)

            # Align to guarantee that the index is identical on both sides.
            # Change the order as needed. Fill the gaps with NaNs.

            # index variables go through an outer join, whereas data variables
            # and non-index coords use an inner join. This avoids creating
            # spurious NaNs in the data variable and only reporting missing
            # elements only once
            lhs, rhs = xarray.align(lhs, rhs, join=join)

            # Build array of bools that highlight all differences, use it to
            # filter the two inputs, and finally convert only the differences
            # to pure python. This is MUCH faster than iterating on all
            # elements in the case where most elements are identical.
            if lhs.dtype.kind in 'iufc' and rhs.dtype.kind in 'iufc':
                # Both arrays are numeric
                # i = int8, int16, int32, int64
                # u = uint8,uint16, uint32, uint64
                # f = float32, float64
                # c = complex64, complex128
                diffs = ~numpy.isclose(
                    lhs.values, rhs.values,
                    rtol=rel_tol, atol=abs_tol, equal_nan=True)

            elif lhs.dtype.kind == 'M' and rhs.dtype.kind == 'M':
                # Both arrays are datetime64
                # Unlike with numpy.isclose(equal_nan=True), there is no
                # straightforward way to do a comparison of dates where
                # NaT == NaT returns True.
                # All datetime64's, including NaT, can be cast to milliseconds
                # since 1970-01-01 (NaT is a special harcoded value).
                # We must first normalise the subtype, so that you can
                # transparently compare e.g. <M8[ns] vs. <M8[D]
                diffs = (lhs.astype('<M8[ns]').astype(int)
                         != rhs.astype('<M8[ns]').astype(int))

            else:
                # At least one between lhs and rhs is non-numeric,
                # e.g. bool or str
                diffs = lhs.values != rhs.values

                # Comparison between two non-scalar, incomparable types
                # (like strings and numbers) will return True
                if diffs is True:
                    diffs = numpy.full(lhs.shape, dtype=bool, fill_value=True)

            if diffs.ndim > 1 and lhs.dims[-1] == '__stacked__':
                # N>0 original dimensions, some (but not all) of which are in
                # brief_dims
                assert brief_dims
                # Produce diffs count along brief_dims
                diffs = diffs.astype(int).sum(
                    axis=tuple(range(diffs.ndim - 1)))
                # Reattach original coords
                diffs = xarray.DataArray(
                    diffs, dims=['__stacked__'],
                    coords={'__stacked__': lhs.coords['__stacked__']})
                # Filter out identical elements
                diffs = diffs[diffs != 0]
                # Convert the diff count to plain dict with the original coords
                diffs = _dataarray_to_dict(diffs)
                for k, count in sorted(diffs.items()):
                    yield diff("%d differences" % count, print_path=path + [k])

            elif '__stacked__' not in lhs.dims:
                # N>0 original dimensions, all of which are in brief_dims

                # Produce diffs count along brief_dims
                count = diffs.astype(int).sum()
                if count:
                    yield diff("%d differences" % count)
            else:
                # N>0 original dimensions, none of which are in brief_dims

                # Filter out identical elements
                lhs = lhs[diffs]
                rhs = rhs[diffs]
                # Convert the original arrays to plain dict
                lhs = _dataarray_to_dict(lhs)
                rhs = _dataarray_to_dict(rhs)

                if join == 'outer':
                    # We're here showing the differences of two non-range
                    # indices, aligned on themselves. All dict values are NaN
                    # by definition, so we can print a terser output by
                    # converting the dicts to sets.
                    lhs = {k for k, v in lhs.items() if not pandas.isnull(v)}
                    rhs = {k for k, v in rhs.items() if not pandas.isnull(v)}

                # Finally dump out all the differences
                yield from _recursive_diff(
                    lhs, rhs, rel_tol=rel_tol, abs_tol=abs_tol,
                    brief_dims=brief_dims, path=path,
                    suppress_type_diffs=True, join=join)

        else:
            # 0-dimensional arrays
            assert lhs.dims == ()
            assert rhs.dims == ()
            yield from _recursive_diff(
                lhs.values.tolist(), rhs.values.tolist(),
                rel_tol=rel_tol, abs_tol=abs_tol, brief_dims=brief_dims,
                path=path, suppress_type_diffs=True, join=join)

    else:
        # unknown objects
        try:
            if lhs != rhs:
                yield diff('%s != %s' % (lhs_repr, rhs_repr))
        except Exception:
            # e.g. bool(xarray.DataArray([1, 2]) == {1: 2}) will raise:
            #   ValueError: The truth value of an array with more than one
            #   element is ambiguous. Use a.any() or a.all()
            # Note special case of comparing an array vs. a list is handled
            # above in this function.
            # Custom classes which implement a duck-typed __eq__ will
            # possibly fail with AttributeError, IndexError, etc. instead.
            yield diff("Cannot compare objects: %s, %s" % (lhs_repr, rhs_repr))


def _str_trunc(x):
    """Helper function of :func:`recursive_diff`.

    Convert x to string. If it is longer than 80 characters, or spans
    multiple lines, truncate it
    """
    x = str(x)
    if len(x) <= 80 and '\n' not in x:
        return x
    return x.splitlines()[0][:76] + ' ...'


@singledispatch
def cast(obj, brief_dims):
    """Helper function of :func:`recursive_diff`.

    Cast objects into simpler object types:

    - Cast tuple to list
    - Cast frozenset to set
    - Cast all numpy-based objects to :class:`xarray.DataArray`, as it is the
      most generic format that can describe all use cases:

      - :class:`numpy.ndarray`
      - :class:`pandas.Series`
      - :class:`pandas.DataFrame`
      - :class:`pandas.Index`, except :class:`pandas.RangeIndex`, which is
        instead returned unaltered
      - :class:`xarray.Dataset`

    The data will be potentially wrapped by a dict to hold the various
    attributes and marked so that it doesn't trigger an infinite recursion.

    - Do nothing for any other object types.

    :param obj:
        complex object that must be simplified
    :param tuple brief_dims:
        sequence of xarray dimensions that must be compacted.
        See documentation on :func:`recursive_diff`.
    :returns:
        simpler object to compare

    **Custom objects**

    This is a single dispatch function which can be extended to compare
    custom objects. Take for example this custom class::

        >>> class Rectangle:
        ...    def __init__(self, w, h):
        ...        self.w = w
        ...        self.h = h
        ...
        ...    def __eq__(self, other):
        ...        return self.w == other.w and self.h == other.h
        ...
        ...    def __repr__(self):
        ...        return 'Rectangle(%f, %f)' % (self.w, self.h)

    The above can be processed by recursive_diff, because it supports the ==
    operator against objects of the same type, and when converted to string
    it conveys meaningful information::

        >>> list(recursive_diff(Rectangle(1, 2), Rectangle(3, 4)))
        ['Rectangle(1.000000, 2.000000) != Rectangle(2.000000, 3.000000)']

    However, it doesn't support the more powerful features of recursive_diff,
    namely recursion and tolerance:

        >>> list(recursive_diff(
        ...     Rectangle(1, 2), Rectangle(1.1, 2.2), abs_tol=.5))
        ['Rectangle(1.0000000, 2.0000000) != Rectangle(1.100000, 2.200000)']

    This can be fixed by registering a custom cast function::

        >>> @cast.register(Rectangle)
        ... def _(obj, brief_dims):
        ...     return {'w': obj.w, 'h': obj.h}

    After doing so, w and h will be compared with tolerance and, if they are
    collections, will be recursively descended into::

        >>> list(recursive_diff(
        ...     Rectangle(1, 2), Rectangle(1.1, 2.7), abs_tol=.5))
        ['[h]: 2.0 != 2.7 (abs: 7.0e-01, rel: 3.5e-01)']
    """
    # This is a single dispatch function, defining the default for any
    # classes not explicitly registered below.
    return obj


@cast.register(numpy.integer)
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for all numpy scalar
    integers (not to be confused with numpy arrays of integers)
    """
    return int(obj)


@cast.register(numpy.floating)
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for all numpy scalar
    floats (not to be confused with numpy arrays of floats)
    """
    return float(obj)


@cast.register(numpy.ndarray)
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`numpy.ndarray`.

    Map to a DataArray with dimensions dim_0, dim_1, ... and
    RangeIndex() as the coords.
    """
    data = _strip_dataarray(xarray.DataArray(obj), brief_dims=brief_dims)
    obj = {
        'dim_%d' % i: pandas.RangeIndex(size)
        for i, size in enumerate(obj.shape)
    }
    obj['data'] = data
    return obj


@cast.register(pandas.Series)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`pandas.Series`.

    Map to a DataArray.
    """
    return {
        'name': obj.name,
        'data': _strip_dataarray(
            xarray.DataArray(obj, dims=['index']), brief_dims=brief_dims),
        'index': obj.index,
    }


@cast.register(pandas.DataFrame)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`pandas.DataFrame`.

    Map to a DataArray.

    TODO: proper support for columns with different dtypes. Right now
    they are cast to the closest common type by DataFrame.values.
    """
    return {
        'data': _strip_dataarray(
            xarray.DataArray(obj, dims=['index', 'column']),
            brief_dims=brief_dims),
        'index': obj.index,
        'columns': obj.columns,
    }


@cast.register(xarray.DataArray)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`xarray.DataArray`.

    Map to a simpler DataArray, with separate indices, non-index coords,
    name, and attributes.
    """
    # Prevent infinite recursion - see _strip_dataarray()
    if '__strip_dataarray__' in obj.attrs:
        return obj

    # Strip out the non-index coordinates and attributes
    return {
        'name': obj.name,
        'attrs': obj.attrs,
        # Index is handled separately, and created as a default
        # RangeIndex(shape[i]) if it doesn't exist, as it is compared
        # with outer join, whereas non-index coords and data are
        # compared with inner joinu
        'index': {
            k: obj.coords[k].to_index()
            for k in obj.dims
        },
        'coords': {
            k: _strip_dataarray(v, brief_dims=brief_dims)
            for k, v in obj.coords.items()
            if not isinstance(v.variable, xarray.IndexVariable)
        },
        'data': _strip_dataarray(obj, brief_dims=brief_dims)
    }


@cast.register(xarray.Dataset)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`xarray.Dataset`.

    Map to a dict of DataArrays.
    """
    return {
        'attrs': obj.attrs,
        # There may be coords, index or not, that are not
        # used in any data variable.
        # See above on why indices are handled separately
        'index': {
            k: obj.coords[k].to_index()
            for k in obj.dims
        },
        'coords': {
            k: _strip_dataarray(v, brief_dims=brief_dims)
            for k, v in obj.coords.items()
            if not isinstance(v.variable, xarray.IndexVariable)
        },
        'data_vars': {
            k: _strip_dataarray(v, brief_dims=brief_dims)
            for k, v in obj.data_vars.items()
        }
    }


@cast.register(pandas.MultiIndex)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`pandas.MultiIndex`.

    Map to a set of tuples. Note that this means that levels are
    positional. Using a set allows comparing the indices non-positionally.
    """
    return {
        'names': obj.names,
        'data': set(obj.tolist())
    }


@cast.register(pandas.RangeIndex)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`pandas.RangeIndex`.

    This function does nothing - RangeIndex objects are dealt with
    directly by :func:`_recursive_diff`. This function is defined
    to prevent RangeIndex objects to be processed by the more generic
    ``cast(obj: pandas.Index)`` below.
    """
    return obj


@cast.register(pandas.Index)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`pandas.Index`.

    Cast to a DataArray.

    .. note::
       :func:`~functools.singledispatch` always prefers a more specialised
       variant if available, so this function will not be called for
       :class:`pandas.MultiIndex` or :class:`pandas.RangeIndex`, as they have
       their own single dispatch variants.
    """
    return _strip_dataarray(xarray.DataArray(obj), brief_dims=brief_dims)


@cast.register(frozenset)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`frozenset`.

    Cast to a set.
    """
    return set(obj)


@cast.register(tuple)  # noqa:F811
def _(obj, brief_dims):
    """Single dispatch specialised variant of :func:`cast` for
    :class:`tuple`.

    Cast to a list.
    """
    return list(obj)


def _strip_dataarray(obj, brief_dims):
    """Helper function of :func:`recursive_diff`.

    Analyse a :class:`xarray.DataArray` and:

    - strip away any non-index coordinates (including scalar coords)
    - create stub coords for dimensions without coords
    - sort dimensions alphabetically
    - ravel the array to a 1D array with (potentially) a MultiIndex.
      brief_dims, if any, are excluded.

    :param obj:
        any xarray.DataArray
    :param brief_dims:
        sequence of dims, or "all"
    :returns:
        a stripped-down shallow copy of obj; otherwise None
    """
    res = obj.copy()

    # Remove non-index coordinates
    for k, v in obj.coords.items():
        if not isinstance(v.variable, xarray.IndexVariable):
            del res[k]

    # Ravel the array to make it become 1-dimensional.
    # To do this, we must first unstack any already stacked dimension.
    for dim in obj.dims:
        if isinstance(obj.get_index(dim), pandas.MultiIndex):
            res = proper_unstack(res, dim)

    # Transpose to ignore dimensions order
    res = res.transpose(*sorted(res.dims))

    # Finally stack everything back together
    if brief_dims != "all":
        stack_dims = sorted(set(res.dims) - set(brief_dims))
        if stack_dims:
            res = res.stack(__stacked__=stack_dims)

    # Prevent infinite recursion - see cast(obj: xarray.DataArray)
    res.attrs['__strip_dataarray__'] = True
    return res


def _get_stripped_dims(a):
    """Helper function of :func:`recursive_diff`.

    :param xarray.DataArray a:
        array that has been stripped with :func:`_strip_dataarray`
    :returns:
        list of original dims, sorted alphabetically
    """
    if '__stacked__' in a.dims:
        res = set(a.coords['__stacked__'].to_index().names)
        res |= set(a.dims) - set(['__stacked__'])
        return sorted(res)
    return list(a.dims)


def _dtype_str(obj):
    """Generate dtype information for object.
    For non-numpy objects, this is just the object class.
    Numpy-based objects also contain the data type (e.g. int32).

    Fixed-length numpy strings that differ only by length should be
    treated as identical, e.g. <U3 and <U6 will both return <U.
    Sub-types of datetime64 must also be discarded.

    :param obj:
        any object being compared
    :return:
        dtype string
    """
    try:
        dtype = type(obj).__name__
    except AttributeError:
        # Base types don't have __name__
        dtype = str(type(obj))

    if isinstance(obj, numpy.integer):
        dtype = 'int'
    elif isinstance(obj, numpy.floating):
        dtype = 'float'

    if isinstance(obj, (numpy.ndarray, pandas.Series, xarray.DataArray)):
        np_dtype = obj.dtype
    elif isinstance(obj, pandas.DataFrame):
        # TODO: support for DataFrames with different dtypes on different
        # columns. See also cast(obj: pandas.DataFrame)
        np_dtype = obj.values.dtype
    else:
        np_dtype = None

    if np_dtype:
        np_dtype = str(np_dtype)
        if np_dtype[:2] in {'<U', '|S'}:
            np_dtype = np_dtype[:2] + '...'
        if np_dtype.startswith('datetime64'):
            np_dtype = 'datetime64'
        return '%s<%s>' % (dtype, np_dtype)
    return dtype


def _dataarray_to_dict(a):
    """Helper function of :func:`recursive_diff`.
    Convert a DataArray prepared by :func:`_strip_dataarray` to a plain
    Python dict.

    :param a:
        :class:`xarray.DataArray` which has exactly 1 dimension,
        no non-index coordinates, and a MultiIndex on its dimension.
    :returns:
        Plain python dict, where the keys are a string representation
        of the points of the MultiIndex.

    .. note::
       Order will be discarded. Duplicate coordinates are not supported.
    """
    assert a.dims == ('__stacked__', )
    dims = a.coords['__stacked__'].to_index().names
    res = {}
    for idx, val in a.to_pandas().iteritems():
        key = ', '.join('%s=%s' % (d, i) for d, i in zip(dims, idx))
        # Prettier output when there was no coord at the beginning,
        # e.g. with plain numpy arrays
        key = re.sub(r'dim_\d+=', '', key)
        res[key] = val
    return res
