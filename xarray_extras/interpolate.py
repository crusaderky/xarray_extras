"""xarray interpolation functions

.. codeauthor:: Guido Imperiale
"""
import xarray
import numpy as np
from xarray.core.pycompat import dask_array_type
from scipy.interpolate import make_interp_spline, BSpline


__all__ = ('splrep', 'splev')


def splrep(a, dim, k=3):
    """Calculate the univariate B-spline for an N-dimensional array

    :param xarray.DataArray a:
        any :class:`~xarray.DataArray`
    :param dim:
        dimension of a to be interpolated
    :param int k:
        B-spline order:

        = ==================
        k interpolation kind
        = ==================
        0 nearest neighbour
        1 linear
        2 quadratic
        3 cubic
        = ==================

    :returns:
        :class:`~xarray.Dataset` with t, c, k (knots, coefficients, order)
        variables, the same shape and coords as the input, that can be passed
        to :func:`splev`.

    Example::

        >>> x = np.arange(0, 120, 20)
        >>> x = xarray.DataArray(x, dims=['x'], coords={'x': x})
        >>> s = xarray.DataArray(np.linspace(1, 20, 5), dims=['s'])
        >>> y = np.exp(-x / s)
        >>> x_new = np.arange(0, 120, 1)
        >>> tck = splrep(y, 'x')
        >>> y_new = splev(x_new, tck)

    **Features**

    - Interpolate a ND array on any arbitrary dimension
    - dask supported on both on the interpolated array and x_new
    - Supports ND x_new arrays
    - The CPU-heavy interpolator generation (:func:`splrep`) is executed only
      once and then can be applied to multiple x_new (:func:`splev`)
    - memory-efficient
    - Can be pickled and used on dask distributed

    **Limitations**

    - Chunks are not supported along dim on the interpolated dimension.
    """
    # Make sure that dim is monotonic ascending and is on axis 0
    a = a.sortby(dim)
    a = a.transpose(dim, *[d for d in a.dims if d != dim])

    if k > 1:
        # If any series contains NaNs, all (and only) the y_new along that
        # series will be NaN.
        # See why  scipy.interpolate.interp1d does it wrong:
        # https://github.com/scipy/scipy/issues/8781
        # https://github.com/scipy/scipy/blob/v1.0.0/scipy/interpolate/interpolate.py#L504
        na_mask = ~a.isnull().any(dim)
        a = a.fillna(1)
    else:
        na_mask = None

    x = a.coords[dim].values

    if x.dtype.kind == 'M':
        # Same treatment will be applied to x_new.
        # Allow x_new.dtype==M8[D] and x.dtype==M8[ns], or vice versa
        # Don't use int - see https://github.com/scipy/scipy/issues/8782
        x = x.astype('M8[ns]').astype(float)

    if isinstance(a.data, dask_array_type):
        from dask.array import map_blocks, Array
        if len(a.data.chunks[0]) > 1:
            raise NotImplementedError(
                "Unsupported: multiple chunks on interpolation dim")

        spline = map_blocks(
            make_interp_spline,
            x, a.data, k=k, check_finite=False, dtype=float)
        t_name = spline.name + '_t'
        spline_first_key = (spline.name, ) + (0, ) * spline.ndim
        t = Array({(t_name, 0): (getattr, spline_first_key, 't')},
                  name=t_name, dtype=float,
                  chunks=((x.size + k + 1, ), ))
        c = map_blocks(getattr, spline, 'c',
                       name=spline.name + '_c', dtype=float)
    else:
        spline = make_interp_spline(x, a.data, k=k)
        t = spline.t
        c = spline.c

    tck = xarray.Dataset(
        data_vars={
            't': ('__t__', t),
            'c': (a.dims, c),
        },
        coords=a.coords,
        attrs={
            'spline_dim': dim,
            'k': k,
        })

    if na_mask is not None:
        tck['na_mask'] = na_mask

    return tck


def splev(x_new, tck, extrapolate=True):
    """Evaluate the B-spline generated with :func:`splrep`.

    :param x_new:
        Any :class:`~xarray.DataArray` with any number of dims, not necessarily
        the original interpolation dim.
        Alternatively, it can be any 1-dimensional array-like; it will be
        automatically converted to a :class:`~xarray.DataArray` on the
        interpolation dim.

    :param xarray.Dataset tck:
        As returned by :func:`splrep`.
        It can have been:

        - transposed (not recommended, as performance will
          drop if c is not C-contiguous)
        - sliced, reordered, or (re)chunked, on any
          dim except the interpolation dim
        - computed from dask to numpy backend
        - round-tripped to disk

    :param extrapolate:
        True
            Extrapolate the first and last polynomial pieces of b-spline
            functions active on the base interval
        False
            Return NaNs outside of the base interval
        'periodic'
            Periodic extrapolation is used
        'clip'
            Return y[0] and y[-1] outside of the base interval

    :returns:
        :class:`~xarray.DataArray` with all dims of the interpolated array,
        minus the interpolation dim, plus all dims of x_new

    See :func:`splrep` for usage example.
    """
    # Pre-process x_new into a DataArray
    if not isinstance(x_new, xarray.DataArray):
        if not isinstance(x_new, dask_array_type):
            x_new = np.array(x_new)
        if x_new.ndim == 0:
            dims = []
        elif x_new.ndim == 1:
            dims = [tck.spline_dim]
        else:
            raise ValueError("N-dimensional x_new is only supported if "
                             "x_new is a DataArray")
        x_new = xarray.DataArray(x_new, dims=dims,
                                 coords={tck.spline_dim: x_new})

    invalid_dims = {*x_new.dims} & {*tck.c.dims} - {tck.spline_dim}
    if invalid_dims:
        raise ValueError("Overlapping dims between interpolated "
                         "array and x_new: %s" % ",".join(invalid_dims))

    if tck.t.shape != (tck.c.sizes[tck.spline_dim] + tck.k + 1, ):
        raise ValueError("Interpolated dimension has been sliced")

    if x_new.dtype.kind == 'M':
        # Don't use int - see https://github.com/scipy/scipy/issues/8782
        # Note that we're modifying the x_new values, not the x_new coords
        x_new = x_new.astype('M8[ns]').astype(float)

    if extrapolate == 'clip':
        x = tck.coords[tck.spline_dim].values
        if x.dtype.kind == 'M':
            x = x.astype('M8[ns]').astype(float)
        x_new = np.clip(x_new, x[0].tolist(), x[-1].tolist())
        extrapolate = False

    c = tck.c.rename({tck.spline_dim: '__c__'})
    c = c.transpose('__c__', *[dim for dim in c.dims if dim != '__c__'])

    if any(isinstance(v.data, dask_array_type) for v in (x_new, tck.t, c)):
        if tck.t.chunks and len(tck.t.chunks[0]) > 1:
            raise NotImplementedError(
                "Unsupported: multiple chunks on interpolation dim")
        if c.chunks and len(c.chunks[0]) > 1:
            raise NotImplementedError(
                "Unsupported: multiple chunks on interpolation dim")

        from dask.array import atop
        # omitting t and c
        x_new_axes = 'abdefghijklm'[:x_new.ndim]
        c_axes = 'nopqrsuvwxyz'[:c.ndim - 1]
        y_new = atop(splev_kernel,
                     x_new_axes + c_axes,
                     x_new.data, x_new_axes,
                     tck.t.data, 't',
                     c.data, 'c' + c_axes,
                     k=tck.k, extrapolate=extrapolate,
                     concatenate=True, dtype=float)
    else:
        y_new = splev_kernel(x_new.values, tck.t.values, tck.c.values, tck.k,
                             extrapolate=extrapolate)

    y_new = xarray.DataArray(
        y_new, dims=x_new.dims + c.dims[1:],
        coords=x_new.coords)
    y_new.coords.update({
        k: c
        for k, c in c.coords.items()
        if '__c__' not in c.dims})

    # Reinstate NaNs that had been replaced with stubs
    if tck.k > 1:
        y_new = y_new.where(tck.na_mask, np.nan)
    return y_new


def splev_kernel(x_new, t, c, k, extrapolate):
    # spline = BSpline(t, c, k, axis=0, extrapolate=extrapolate)
    spline = BSpline.construct_fast(t, c, k, axis=0, extrapolate=extrapolate)
    return spline(x_new)
