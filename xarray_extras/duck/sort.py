"""Helper functions for :mod:`xarray_extras.sort`, which accept either
numpy arrays or dask arrays.
"""
import dask.array as da
import numpy as np
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

    # Preprocess data, by putting it together with its indexes in a recarray
    # np.core.records.fromarrays won't work if a and idx don't have the same
    # shape
    idx = np.arange(a.shape[-1], dtype=np.int64)
    idx = idx[(np.newaxis, ) * (a.ndim - 1)]

    rec = np.recarray(a.shape, dtype=[('values', a.dtype), ('idx', idx.dtype)])
    rec.values = a
    rec.idx = idx

    return topk(rec, k).idx


def take_along_axis(a, ind):
    """Easily use the outputs of argsort on ND arrays to pick the results.
    """
    if isinstance(a, np.ndarray) and isinstance(ind, np.ndarray):
        return backport_numpy.take_along_axis(a, ind)

    # This is going to be an ugly and slow mess, as dask does not support
    # fancy indexing at all; also selection by dask arrays of ints has not
    # been merged yet.

    assert a.shape[:-1] == ind.shape[:-1]
    final_shape = ind.shape
    ind = ind.reshape(ind.size // ind.shape[-1], ind.shape[-1])
    a = a.reshape(a.size // a.shape[-1], a.shape[-1])
    res = []

    for a_i, ind_i in zip(a, ind):
        if not isinstance(a_i, da.Array):
            a_i = da.from_array(a_i, chunks=a_i.shape)

        if isinstance(ind_i, da.Array):
            res_i = backport_dask.slice_with_int_dask_array_on_axis(
                a_i, ind_i, axis=0)
        else:
            res_i = a_i[ind_i]
        res.append(res_i)

    res = da.stack(res)
    return res.reshape(*final_shape)
