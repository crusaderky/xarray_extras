"""Sorting functions"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TypeVar

import xarray

from xarray_extras.duck import sort as duck

__all__ = ("argtopk", "take_along_dim", "topk")


T = TypeVar("T", xarray.DataArray, xarray.Dataset)
TV = TypeVar("TV", xarray.DataArray, xarray.Dataset, xarray.Variable)


def topk(a: TV, k: int, dim: Hashable, split_every: int | None = None) -> TV:
    """Extract the k largest elements from a on the given dimension, and return
    them sorted from largest to smallest. If k is negative, extract the -k
    smallest elements instead, and return them sorted from smallest to largest.

    This assumes that ``k`` is small.  All results will be returned in a single
    chunk along the given axis.
    """
    return xarray.apply_ufunc(
        duck.topk,
        a,
        kwargs={"k": k, "split_every": split_every},
        input_core_dims=[[dim]],
        output_core_dims=[["__temp_topk__"]],
        dask="allowed",
    ).rename({"__temp_topk__": dim})


def argtopk(a: TV, k: int, dim: Hashable, split_every: int | None = None) -> TV:
    """Extract the indexes of the k largest elements from a on the given
    dimension, and return them sorted from largest to smallest. If k is
    negative, extract the -k smallest elements instead, and return them
    sorted from smallest to largest.

    This assumes that ``k`` is small.  All results will be returned in a single
    chunk along the given axis.
    """
    return xarray.apply_ufunc(
        duck.argtopk,
        a,
        kwargs={"k": k, "split_every": split_every},
        input_core_dims=[[dim]],
        output_core_dims=[["__temp_topk__"]],
        dask="allowed",
    ).rename({"__temp_topk__": dim})


def take_along_dim(a: T, ind: T, dim: Hashable) -> T:
    """Use the output of :func:`argtopk` to pick points from a.

    :param a:
        xarray.DataArray or xarray.Dataset
    :param ind:
        array of ints, as returned by :func:`argtopk`
    :param dim:
        dimension along which argtopk was executed
    """
    a = a.rename({dim: "__temp_take_along_dim__"})

    return xarray.apply_ufunc(
        duck.take_along_axis,
        a,
        ind,
        input_core_dims=[["__temp_take_along_dim__"], [dim]],
        output_core_dims=[[dim]],
        dask="allowed",
    )
