"""Advanced cumulative sum/productory/mean functions
"""
from typing import Callable, Hashable, Optional, TypeVar
import dask.array as da
import numpy as np
import xarray

from .kernels import cumulatives as kernels


__all__ = ('cummean', 'compound_sum', 'compound_prod', 'compound_mean')


T = TypeVar('T', xarray.DataArray, xarray.Dataset)
TV = TypeVar('TV', xarray.DataArray, xarray.Dataset, xarray.Variable)


def cummean(x: TV, dim: Hashable, skipna: Optional[bool] = None) -> TV:
    """
    .. math::

        y_{i} = mean(x_{0}, x_{1}, ... x_{i})

    :param x:
        :class:`xarray.DataArray`, :class:`xarray.Dataset`, or
        :class:`xarray.Variable`
    :param hashable dim:
        dimension along which to calculate the mean
    :param bool skipna:
        If True, skip missing values (as marked by NaN). By default, only skips
        missing values for float dtypes; other dtypes either do not have a
        sentinel missing value (int) or skipna=True has not been implemented
        (object, datetime64 or timedelta64).
    :returns:
        xarray object of the same type, dtype, and shape as x
    """
    if skipna is False or (skipna is None and x.dtype.kind not in 'fc'):
        # n is a simple arange
        if x.chunks:
            n = da.arange(1, x.sizes[dim] + 1,
                          chunks=x.chunks[x.dims.index(dim)])
        else:
            n = np.arange(1, x.sizes[dim] + 1)
        n = xarray.DataArray(n, dims=[dim], coords={dim: x.coords[dim]})
    else:
        # heavier computation
        n = (~x.isnull()).cumsum(dim, skipna=False)

    return x.cumsum(dim, skipna=skipna) / n


def compound_sum(x: T, c: xarray.DataArray, xdim: Hashable, cdim: Hashable
                 ) -> T:
    """Compound sum on arbitrary points of x along dim.

    :param x:
        :class:`xarray.DataArray` or :class:`xarray.Dataset` containing the
        data to be compounded
    :param xarray.DataArray c:
        array where every row contains elements of x.coords[xdim] and
        is used to build a point of the output.
        The cells in the row are matched against x.coords[dim] and perform a
        sum. If different rows of c require different amounts of points from x,
        they must be padded on the right with NaN, NaT, or '' (respectively for
        numbers, datetimes, and strings).
    :param hashable xdim:
        dimension of x to acquire data from. The coord associated to it must be
        monotonic ascending.
    :param hashable cdim:
        dimension of c that represent the vector of points to be compounded for
        every point of dim
    :returns:
        xarray object of the same type and dtype as x, with all dims from x
        and c except xdim and cdim.

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


def compound_prod(x: T, c: xarray.DataArray, xdim: Hashable, cdim: Hashable
                  ) -> T:
    """Compound product among arbitrary points of x along dim
    See :func:`compound_sum`.
    """
    return _compound(x, c, xdim, cdim, kernels.compound_prod)


def compound_mean(x: T, c: xarray.DataArray, xdim: Hashable, cdim: Hashable
                  ) -> T:
    """Compound mean among arbitrary points of x along dim
    See :func:`compound_sum`.
    """
    return _compound(x, c, xdim, cdim, kernels.compound_mean)


def _compound(x: T, c: xarray.DataArray, xdim: Hashable, cdim: Hashable,
              kernel: Callable[[T, xarray.DataArray], T]) -> T:
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
