"""dask kernels for :mod:`xarray_extras.interpolate`

.. codeauthor:: Guido Imperiale
"""
from scipy.interpolate import BSpline


def splev(x_new, t, c, k, extrapolate):
    """Generate a BSpline object on the fly from knots and coefficients and
    evaluate it on x_new.

    See :class:`scipy.interpolate.BSpline` for all parameters.
    """
    spline = BSpline.construct_fast(t, c, k, axis=0, extrapolate=extrapolate)
    return spline(x_new)
