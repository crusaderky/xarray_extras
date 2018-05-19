"""Helper functions for :mod:`xarray_extras.sort`, which accept either
numpy arrays or dask arrays.
"""
import dask.array as da
import numpy as np
from xarray.core.duck_array_ops import broadcast_to
from ..backport import dask as backport_dask
from ..backport import numpy as backport_numpy


def topk(a, k, split_every=None):
    """If a is a :class:`dask.array.Array`, invoke a.topk; else reimplement
    the functionality in plain numpy.
    """
    if isinstance(a, da.Array):
        return a.topk(k, split_every=split_every)

    if abs(k) < a.shape[-1]:
        a = np.partition(a, -k)
        if k > 0:
            a = a[..., -k:]
        else:
            a = a[..., :-k]

    # Sort the partitioned output
    a = np.sort(a)
    if k > 0:
        # Sort from greatest to smallest
        return a[..., ::-1]
    return a


def argtopk(a, k, split_every=None):
    """If a is a :class:`dask.array.Array`, invoke a.argtopk; else reimplement
    the functionality in plain numpy.
    """
    if isinstance(a, da.Array):
        return a.argtopk(k, split_every=split_every)

    idx = np.argpartition(a, -k)
    if k > 0:
        idx = idx[..., -k:]
    else:
        idx = idx[..., :-k]

    a = backport_numpy.take_along_axis(a, idx, axis=-1)
    idx = backport_numpy.take_along_axis(idx, a.argsort(), axis=-1)
    if k > 0:
        # Sort from greatest to smallest
        return idx[..., ::-1]
    return idx


def take_along_axis(a, ind):
    """Easily use the outputs of argsort on ND arrays to pick the results.
    """
    if isinstance(a, np.ndarray) and isinstance(ind, np.ndarray):
        ind = ind.reshape((1, ) * (a.ndim - ind.ndim) + ind.shape)
        res = backport_numpy.take_along_axis(a, ind, axis=-1)
        return res

    # This is going to be an ugly and slow mess, as dask does not support
    # fancy indexing.

    # Normalize a and ind. The end result is that a can have more axes than
    # ind on the left, but not vice versa, and that all axes except the
    # extra ones on the left and the rightmost one (the axis to take
    # along) are the same shape.
    if ind.ndim > a.ndim:
        a = a.reshape((1, ) * (ind.ndim - a.ndim) + a.shape)
    common_shape = tuple(np.maximum(a.shape[-ind.ndim:-1], ind.shape[:-1]))
    a_extra_shape = a.shape[:-ind.ndim]
    a = broadcast_to(a, a_extra_shape + common_shape + a.shape[-1:])
    ind = broadcast_to(ind, common_shape + ind.shape[-1:])

    # Flatten all common axes onto axis -2
    final_shape = a.shape[:-ind.ndim] + ind.shape
    ind = ind.reshape(ind.size // ind.shape[-1], ind.shape[-1])
    a = a.reshape(*a_extra_shape, ind.shape[0], a.shape[-1])

    # Now we have a[..., i, j] and ind[i, j], where i are the flattened
    # common axes and j is the axis to take along.
    res = []

    # Cycle a and ind along i, perform 1D slices, and then stack them back
    # together
    for i in range(ind.shape[0]):
        a_i = a[..., i, :]
        ind_i = ind[i, :]

        if not isinstance(a_i, da.Array):
            a_i = da.from_array(a_i, chunks=a_i.shape)

        if isinstance(ind_i, da.Array):
            res_i = backport_dask.slice_with_int_dask_array_on_axis(
                a_i, ind_i, axis=a_i.ndim - 1)
        else:
            res_i = a_i[..., ind_i]
        res.append(res_i)

    res = da.stack(res, axis=-2)
    # Un-flatten axis i
    res = res.reshape(*final_shape)
    return res
