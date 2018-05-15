import numpy.core.numeric as _nx
from numpy.core.multiarray import normalize_axis_index
from numpy.core.numeric import asanyarray


def _make_along_axis_idx(arr, indices, axis):
    """Backport from `<https://github.com/numpy/numpy/issues/8714>`
    """
    # compute dimensions to iterate over
    shape_ones = (1,) * indices.ndim
    ins_ndim = indices.ndim - (arr.ndim - 1)  # inserted dimensions
    if ins_ndim < 0:
        raise ValueError("`indices` must have ndim >= arr.ndim - 1")
    dest_dims = list(range(axis)) + [None] + \
        list(range(axis + ins_ndim, indices.ndim))

    # build a fancy index, consisting of orthogonal aranges, with the
    # requested index inserted at the right location
    fancy_index = []
    for dim, n in zip(dest_dims, arr.shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1:]
            fancy_index.append(_nx.arange(n).reshape(ind_shape))

    return tuple(fancy_index)


def take_along_axis(arr, indices, axis):
    """Backport from `<https://github.com/numpy/numpy/issues/8714>`_.

    Take the elements described by `indices` along each 1-D slice of the given
    `axis`, matching up subspaces of arr and indices.
    This function can be used to index with the result of `argsort`, `argmax`,
    and other `arg` functions.
    For each `i...`, `j...`, `k...` in turn (representing a series of indices),
    where the number of indices in `i...` is equal to `axis`, this computes::
        out[i..., j..., k...] = arr[i..., indices[i..., j..., k...], k...]
    Or equivalently (where `...` alone is the builtin `Ellipsis`):
        out[i..., ..., k...] = arr[i..., :, k...][indices[i..., ..., k...]]
    .. versionadded:: 1.13.0
    Parameters
    ----------
    arr: array_like (Ni..., M, Nk...)
        source array
    indices: array_like (Ni..., Nj..., Nk...)
        indices to take along each 1d slice of `arr`
    axis: int
        the axis to take 1d slices along
    Returns
    -------
    out: ndarray (A..., K..., B...)
        The indexed result.
        out[a..., k..., b...] = arr[a..., indices[a..., k..., b...], b...]
    See Also
    --------
    take : Take along an axis without matching up subspaces
    Examples
    --------
    For this sample array
    >>> a = np.array([[10, 30, 20], [60, 40, 50]])
    We can sort either by using sort directly, or argsort and this function
    >>> np.sort(a, axis=1)
    array([[10, 20, 30],
           [40, 50, 60]])
    >>> ai = np.argsort(a, axis=1); ai
    array([[0, 2, 1],
           [1, 2, 0]], dtype=int64)
    >>> np.take_along_axis(a, ai, axis=1)
    array([[10, 20, 30],
           [40, 50, 60]])
    The same works for max and min:
    >>> np.max(a, axis=1)
    array([30, 60])
    >>> ai = np.argmax(a, axis=1); ai
    array([1, 0], dtype=int64)
    >>> np.take_along_axis(a, ai, axis=1)
    array([30, 60])
    If we want to get the max and min at the same time, we can stack the
    indices first
    >>> ai_min = np.argmin(a, axis=1)
    >>> ai_max = np.argmax(a, axis=1)
    >>> ai = np.stack([ai_min, ai_max], axis=axis); ai
    array([[0, 1],
           [1, 0]], dtype=int64)
    >>> np.take_along_axis(a, ai, axis=1)
    array([[10, 30],
           [40, 60]])
    """
    # normalize inputs
    arr = asanyarray(arr)
    indices = asanyarray(indices)
    if axis is None:
        arr = arr.ravel()
        axis = 0
    else:
        axis = normalize_axis_index(axis, arr.ndim)
    if not _nx.issubdtype(indices.dtype, _nx.integer):
        raise IndexError('arrays used as indices must be of integer type')

    # use the fancy index
    return arr[_make_along_axis_idx(arr, indices, axis)]
