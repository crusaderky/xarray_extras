"""Helper functions for :mod:`xarray_extras.sort`, which accept either
numpy arrays or dask arrays.
"""
import dask.array as da
import numpy as np


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
