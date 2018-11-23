"""xarray spline interpolation functions
"""
import xarray
import numpy as np
from xarray.core.pycompat import dask_array_type
from .kernels import interpolate as kernels


__all__ = ('splrep', 'splev')


def splrep(a, dim, k=3):
    """Calculate the univariate B-spline for an N-dimensional array

    :param xarray.DataArray a:
        any :class:`~xarray.DataArray`
    :param dim:
        dimension of a to be interpolated. ``a.coords[dim]`` must be strictly
        monotonic ascending. All int, float (not complex), or datetime dtypes
        are supported.
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
    # Make sure that dim is on axis 0
    a = a.transpose(dim, *[d for d in a.dims if d != dim])
    x = a.coords[dim].values

    if x.dtype.kind == 'M':
        # Same treatment will be applied to x_new.
        # Allow x_new.dtype==M8[D] and x.dtype==M8[ns], or vice versa
        x = x.astype('M8[ns]').astype(float)

    t = kernels.make_interp_knots(x, k, check_finite=False)
    if k < 2:
        t_c_param = None
    else:
        t_c_param = t

    if isinstance(a.data, dask_array_type):
        from dask.array import map_blocks
        if len(a.data.chunks[0]) > 1:
            raise NotImplementedError(
                "Unsupported: multiple chunks on interpolation dim")

        c = map_blocks(
            kernels.make_interp_coeffs,
            x, a.data, k=k, t=t_c_param, check_finite=False, dtype=float)
    else:
        c = kernels.make_interp_coeffs(x, a.data, k=k, t=t_c_param,
                                       check_finite=False)

    return xarray.Dataset(
        data_vars={
            't': ('__t__', t),
            'c': (a.dims, c),
        },
        coords=a.coords,
        attrs={
            'spline_dim': dim,
            'k': k,
        })


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

    dim = tck.spline_dim
    t = tck.t
    c = tck.c
    k = tck.k

    invalid_dims = {*x_new.dims} & {*c.dims} - {dim}
    if invalid_dims:
        raise ValueError("Overlapping dims between interpolated "
                         "array and x_new: %s" % ",".join(invalid_dims))

    if t.shape != (c.sizes[dim] + k + 1, ):
        raise ValueError("Interpolated dimension has been sliced")

    if x_new.dtype.kind == 'M':
        # Note that we're modifying the x_new values, not the x_new coords
        # xarray datetime objects are always in ns
        x_new = x_new.astype(float)

    if extrapolate == 'clip':
        x = tck.coords[dim].values
        if x.dtype.kind == 'M':
            x = x.astype('M8[ns]').astype(float)
        x_new = np.clip(x_new, x[0].tolist(), x[-1].tolist())
        extrapolate = False

    if c.dims[0] != dim:
        c = c.transpose(dim, *[d for d in c.dims if d != dim])

    if any(isinstance(v.data, dask_array_type) for v in (x_new, t, c)):
        if t.chunks and len(t.chunks[0]) > 1:
            raise NotImplementedError(
                "Unsupported: multiple chunks on interpolation dim")
        if c.chunks and len(c.chunks[0]) > 1:
            raise NotImplementedError(
                "Unsupported: multiple chunks on interpolation dim")

        from dask.array import atop
        # omitting t and c
        x_new_axes = 'abdefghijklm'[:x_new.ndim]
        c_axes = 'nopqrsuvwxyz'[:c.ndim - 1]
        y_new = atop(kernels.splev,
                     x_new_axes + c_axes,
                     x_new.data, x_new_axes,
                     t.data, 't',
                     c.data, 'c' + c_axes,
                     k=k, extrapolate=extrapolate,
                     concatenate=True, dtype=float)
    else:
        y_new = kernels.splev(x_new.values, t.values, c.values, k,
                              extrapolate=extrapolate)

    y_new = xarray.DataArray(
        y_new, dims=x_new.dims + c.dims[1:],
        coords=x_new.coords)
    y_new.coords.update({
        k: c
        for k, c in c.coords.items()
        if dim not in c.dims})
    return y_new
