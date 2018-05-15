import numpy as np
from dask.array import Array, atop, from_array


def slice_with_int_dask_array_on_axis(x, idx, axis):
    """Backport from `<https://github.com/dask/dask/pull/3407>`_.

    Slice x with a dask arrays of ints along the given axis

    This is a helper function of `slice_with_int_dask_array`.
    """
    if np.isnan(x.chunks[axis]).any():
        raise NotImplementedError("Slicing an array with unknown chunks with "
                                  "a dask.array of ints is not supported")

    # Calculate the offset at which each chunk starts along axis
    # e.g. chunks=(..., (5, 3, 4), ...) -> offset=[0, 5, 8]
    offset = np.roll(np.cumsum(x.chunks[axis]), 1)
    offset[0] = 0
    offset = from_array(offset, chunks=1)
    # Tamper with the declared chunks of offset to make atop align it with
    # x[axis]
    offset = Array(offset.dask, name=offset.name,
                   chunks=(x.chunks[axis], ), dtype=int)

    # Define axis labels for atop
    x_axes = 'abcdefghijklmnopqrstuvwxy'[:x.ndim]
    idx_axes = 'z'
    offset_axes = x_axes[axis]
    p_axes = x_axes[:axis + 1] + idx_axes + x_axes[axis + 1:]
    y_axes = x_axes[:axis] + idx_axes + x_axes[axis + 1:]

    # Calculate the cartesian product of every chunk of x vs. every chunk of
    # idx
    p = atop(slice_with_int_dask_array_chunk,
             p_axes, x, x_axes, idx, idx_axes, offset, offset_axes,
             axis=axis, dtype=x.dtype)

    # Aggregate on the chunks of x along axis
    y = atop(slice_with_int_dask_array_aggregate,
             y_axes, idx, idx_axes, p, p_axes, concatenate=True,
             x_chunks=x.chunks[axis], axis=axis, dtype=x.dtype)
    return y


def slice_with_int_dask_array_chunk(x, idx, offset, axis):
    """Backport from `<https://github.com/dask/dask/pull/3407>`_.

    Chunk kernel of `slice_with_int_dask_array_on_axis`.

    Returns ``x`` sliced along ``axis``, using only the elements of
    ``idx`` that fall inside the current chunk.
    """
    idx = idx - offset[0]
    idx_filter = np.logical_and(idx >= 0, idx < x.shape[axis])
    idx = idx[idx_filter]
    return x[[
        idx if i == axis else slice(None)
        for i in range(x.ndim)
    ]]


def slice_with_int_dask_array_aggregate(idx, chunk_outputs, x_chunks, axis):
    """Backport from `<https://github.com/dask/dask/pull/3407>`_.

    Final aggregation kernel of `slice_with_int_dask_array_on_axis`.

    Returns ``x`` sliced along ``axis``, using only the elements of
    ``idx`` that fall inside the current chunk.
    """
    offset = 0
    idx_ranges = []
    for x_chunk in x_chunks:
        idx_filter = np.logical_and(idx >= offset, idx < offset + x_chunk)
        idx_ranges.append(np.arange(idx.size)[idx_filter])
        offset += x_chunk
    idx_ranges = np.concatenate(idx_ranges)

    return chunk_outputs[[
        idx_ranges if i == axis else slice(None)
        for i in range(chunk_outputs.ndim)
    ]]
