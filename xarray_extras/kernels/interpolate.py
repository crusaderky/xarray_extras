"""dask kernels for :mod:`xarray_extras.interpolate`

.. codeauthor:: Guido Imperiale
"""
from scipy.interpolate import BSpline


def splev(x_new, t, c, k, extrapolate):
    # spline = BSpline(t, c, k, axis=0, extrapolate=extrapolate)
    spline = BSpline.construct_fast(t, c, k, axis=0, extrapolate=extrapolate)
    return spline(x_new)
