"""Sorting functions
"""
import xarray
from .duck import sort as duck

__all__ = ('topk', 'argtopk', 'take_along_dim')


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


def take_along_dim(a, ind, dim):
    """Use the output of :func:`argtopk` to pick points from a.

    :param a:
        any xarray object
    :param ind:
        array of ints, as returned by :func:`argtopk`
    :param dim:
        dimension along which argtopk was executed
    """
    a = a.rename({dim: dim + '.orig'})

    return xarray.apply_ufunc(
        duck.take_along_axis, a, ind,
        input_core_dims=[[dim + '.orig'], [dim]],
        output_core_dims=[[dim]],
        dask='allowed')
