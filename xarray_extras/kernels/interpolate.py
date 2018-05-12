"""dask kernels for :mod:`xarray_extras.interpolate`

.. codeauthor:: Guido Imperiale
"""
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline
from scipy.interpolate._bsplines import _as_float_array, _not_a_knot, \
    _augknt


def _memoryview_safe(x):
    """Make array safe to run in a Cython memoryview-based kernel. These
    kernels typically break down with the error ``ValueError: buffer source
    array is read-only`` when running in dask distributed.
    """
    if not x.flags.writeable:
        if not x.flags.owndata:
            x = x.copy(order='C')
        x.setflags(write=True)
    return x


def make_interp_knots(x, k=3, bc_type=None, check_finite=True):
    """Compute the knots of the B-spline.

    .. note::
       This is a temporary implementation that should be moved to the main
       scipy library - see `<https://github.com/scipy/scipy/issues/8810>`_.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas.
    k : int, optional
        B-spline degree. Default is cubic, k=3.
    bc_type : 2-tuple or None
        Boundary conditions.
        Default is None, which means choosing the boundary conditions
        automatically. Otherwise, it must be a length-two tuple where the first
        element sets the boundary conditions at ``x[0]`` and the second
        element sets the boundary conditions at ``x[-1]``. Each of these must
        be an iterable of pairs ``(order, value)`` which gives the values of
        derivatives of specified orders at the given edge of the interpolation
        interval.
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default is True.

    Returns
    -------
    numpy array with size = x.size + k + 1, representing the B-spline knots.
    """
    if bc_type is None:
        bc_type = (None, None)

    if k < 2 and bc_type != (None, None):
        raise ValueError("Too much info for k<2: bc_type can only be None.")

    x = np.array(x)
    if x.ndim != 1 or np.any(x[1:] <= x[:-1]):
        raise ValueError("Expect x to be a 1-D sorted array-like.")

    if k == 0:
        t = np.r_[x, x[-1]]
    elif k == 1:
        t = np.r_[x[0], x, x[-1]]
    elif bc_type == (None, None):
        if k == 2:
            # OK, it's a bit ad hoc: Greville sites + omit
            # 2nd and 2nd-to-last points, a la not-a-knot
            t = (x[1:] + x[:-1]) / 2.
            t = np.r_[(x[0],) * (k + 1),
                      t[1:-1],
                      (x[-1],) * (k + 1)]
        else:
            t = _not_a_knot(x, k)
    else:
        t = _augknt(x, k)

    return _as_float_array(t, check_finite)


def make_interp_coeffs(x, y, k=3, t=None, bc_type=None, axis=0,
                       check_finite=True):
    """Compute the knots of the B-spline.

    .. note::
       This is a temporary implementation that should be moved to the main
       scipy library - see `<https://github.com/scipy/scipy/issues/8810>`_.

    See :func:`scipy.interpolate.make_interp_spline` for parameters.

    :param t:
        Knots array, as calculated by :func:`make_interp_knots`.

        - For k=0, must always be None (the coefficients are not a function of
          the knots).
        - For k=1, set to None if t has been calculated by
          :func:`make_interp_knots`; pass a vector if it already existed
          before.
        - For k=2 and k=3, must always pass either the output of
          :func:`make_interp_knots` or a pre-generated vector.
    """
    x = _memoryview_safe(x)
    y = _memoryview_safe(y)
    if t is not None:
        t = _memoryview_safe(t)

    return make_interp_spline(
        x, y, k, t, bc_type=bc_type, axis=axis, check_finite=check_finite).c


def splev(x_new, t, c, k=3, extrapolate=True):
    """Generate a BSpline object on the fly from knots and coefficients and
    evaluate it on x_new.

    See :class:`scipy.interpolate.BSpline` for all parameters.
    """
    t = _memoryview_safe(t)
    c = _memoryview_safe(c)
    x_new = _memoryview_safe(x_new)
    spline = BSpline.construct_fast(t, c, k, axis=0, extrapolate=extrapolate)
    return spline(x_new)
