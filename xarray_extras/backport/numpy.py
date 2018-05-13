import numpy as np


def take_along_axis(a, ind, axis=-1):
    """Easily use the outputs of argsort on ND arrays to pick the results.

    Backport from `<https://github.com/numpy/numpy/issues/8708>`_.

    a: array_like of shape (A..., M, B...)
        source array
    ind: array_like of shape (A..., K..., B...)
        indices to take along each 1d slice of `arr`
    axis: int
        index of the axis with dimension M

    out: array_like of shape (A..., K..., B...)
        out[a..., k..., b...] = arr[a..., inds[a..., k..., b...], b...]
    """
    if axis < 0:
       axis += a.ndim
    ind_shape = (1,) * ind.ndim
    ins_ndim = ind.ndim - (a.ndim - 1)   # inserted dimensions

    dest_dims = list(range(axis)) + [None] + \
        list(range(axis + ins_ndim, ind.ndim))

    inds = []
    for dim, n in zip(dest_dims, a.shape):
        if dim is None:
            inds.append(ind)
        else:
            ind_shape_dim = ind_shape[:dim] + (-1,) + ind_shape[dim + 1:]
            inds.append(np.arange(n).reshape(ind_shape_dim))

    return a[tuple(inds)]