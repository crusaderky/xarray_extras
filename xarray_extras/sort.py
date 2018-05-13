"""xarray sorting functions
"""
import dask.array as da
import numpy as np
import xarray


__all__ = ('topk', 'argtopk')


def topk(a, k, dim, split_every=None):
    """Extract the k largest elements from a on the given dimension, and return
    them sorted from largest to smallest. If k is negative, extract the -k
    smallest elements instead, and return them sorted from smallest to largest.

    This assumes that ``k`` is small.  All results will be returned in a single
    chunk along the given axis.
    """
    return xarray.apply_ufunc(
        _duck_topk, a,
        kwargs={'k': k, 'split_every': split_every},
        input_core_dims=[[dim]],
        output_core_dims=[[dim + '.topk']],
        dask='allowed').rename({dim + '.topk': dim})


def argtopk(a, k, dim, split_every=None):
    """Extract the indexes of the k largest elements from a on the given
    dimension, and return them sorted from largest to smallest. If k is
    negative, extract the -k smallest elements instead, and return them
    sorted from smallest to largest.

    This assumes that ``k`` is small.  All results will be returned in a single
    chunk along the given axis.
    """
    return xarray.apply_ufunc(
        _duck_argtopk, a,
        kwargs={'k': k, 'split_every': split_every},
        input_core_dims=[[dim]],
        output_core_dims=[[dim + '.topk']],
        dask='allowed').rename({dim + '.topk': dim})


def _duck_topk(a, k, split_every=None):
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


def _duck_argtopk(a, k, split_every=None):
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

    return _duck_topk(rec, k).idx
