#!/usr/bin/env python

import xarray

from .kernels import cumulatives as kernels

__all__ = ('cumulative_sum', 'cumulative_prod', 'cumulative_mean',
           'compound_sum', 'compound_prod', 'compound_mean')


def cumulative_sum(x, dim):
    """
    .. math::

        y_{i} = \sum_{j=0}^{\i} x_{j}

    Which is equivalent to
    .. math::

       \[
       \left \{
          y_{0} = x_{0}
          y_{i} = y{i-1} + x{i}
       \right \}
       \]
    """
    return _cumulative(x, dim, kernels.cumulative_sum)


def cumulative_prod(x, dim):
    """
    .. math::

        y_{i} = \prod_{j=0}^{\i} x_{j}

    Which is equivalent to
    .. math::

       \[
       \left \{
          y_{0} = x_{0}
          y_{i} = y{i-1} * x{i}
       \right \}
       \]
    """
    return _cumulative(x, dim, kernels.cumulative_prod)


def cumulative_mean(x, dim):
    """
    .. math::

        y_{i} = mean(x_{0}, x_{1}, ... x_{i})
    """
    return _cumulative(x, dim, kernels.cumulative_mean)


def _cumulative(x, dim, kernel):
    """Implementation of all cumulative functions

    :param kernel:
        numba kernel to apply to x
    """
    return xarray.apply_ufunc(
        kernel, x,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        dask='parallelized',
        output_dtypes=[x.dtype])


def compound_sum(x, c, xdim, cdim):
    """Compound sum on arbitrary points of x along dim.

    :param x:
        Any xarray object containing the data to be compounded
    :param DataArray c:
        array where every row contains elements of x.coords[xdim] and
        is used to build a point of the output.
        The cells in the row are matched against x.coords[dim] and perform a
        sum. If different rows of c require different amounts of points from x,
        they must be padded on the right with NaN, NaT, or '' (respectively for
        numbers, datetimes, and strings).
    :param str xdim:
        dimension of x to acquire data from. The coord associated to it must be
        monotonic ascending.
    :param str cdim:
        dimension of c that represent the vector of points to be compounded for
        every point of dim
    :returns:
        DataArray with all dims from x and c, except xdim and cdim, and the
        same dtype as x.

    example::

        >>> x = xarray.DataArray(
        >>>     [10, 20, 30],
        >>>     dims=['x'], coords={'x': ['foo', 'bar', 'baz']})
        >>> c = xarray.DataArray(
        >>>     [['foo', 'baz', None],
        >>>      ['bar', 'baz', 'baz']],
        >>>      dims=['y', 'c'], coords={'y': ['new1', 'new2']})
        >>> compound_sum(x, c, 'x', 'c')
        <xarray.DataArray (y: 2)>
        array([40, 80])
        Coordinates:
          * y        (y) <U4 'new1' 'new2'
    """
    return _compound(x, c, xdim, cdim, kernels.compound_sum)


def compound_prod(x, c, xdim, cdim):
    """Compound product among arbitrary points of x along dim
    See :func:`compound_sum`.
    """
    return _compound(x, c, xdim, cdim, kernels.compound_prod)


def compound_mean(x, c, xdim, cdim):
    """Compound mean among arbitrary points of x along dim
    See :func:`compound_sum`.
    """
    return _compound(x, c, xdim, cdim, kernels.compound_mean)


def _compound(x, c, xdim, cdim, kernel):
    """Implementation of all compound functions

    :param kernel:
        numba kernel to apply to (x, idx), where
        idx is an array of indices with the same shape as c,
        containing the indices along x.coords[xdim] or -1 where c is null.
    """
    # Convert coord points to indexes of x.coords[dim]
    idx = xarray.DataArray(
        x.coords[xdim].searchsorted(c),
        dims=c.dims, coords=c.coords)
    # searchsorted(NaN) returns 0; replace it with -1.
    # isnull('') returns False. We could have asked for None, however
    # searchsorted will refuse to compare strings and None's
    if c.dtype.kind == 'U':
        idx = idx.where(c != '', -1)
    else:
        idx = idx.where(~c.isnull(), -1)

    dtype = x.dtypes if isinstance(x, xarray.Dataset) else x.dtype

    return xarray.apply_ufunc(
        kernel, x, idx,
        input_core_dims=[[xdim], [cdim]],
        output_core_dims=[[]],
        dask='parallelized',
        output_dtypes=[dtype])
