"""xarray sorting functions
"""
import xarray
from .duck import sort as duck

__all__ = ('topk', 'argtopk')


def topk(a, k, dim, split_every=None):
    """Extract the k largest elements from a on the given dimension, and return
    them sorted from largest to smallest. If k is negative, extract the -k
    smallest elements instead, and return them sorted from smallest to largest.

    This assumes that ``k`` is small.  All results will be returned in a single
    chunk along the given axis.
    """
    return xarray.apply_ufunc(
        duck.topk, a,
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
        duck.argtopk, a,
        kwargs={'k': k, 'split_every': split_every},
        input_core_dims=[[dim]],
        output_core_dims=[[dim + '.topk']],
        dask='allowed').rename({dim + '.topk': dim})
