import numpy as np


def slice_with_int_dask_array_on_axis(x, idx, axis):
    """Backport from `<https://github.com/dask/dask/pull/3407>`_.

    Slice x with a dask arrays of ints along the given axis

    This is a helper function of `slice_with_int_dask_array`.
    """
    from dask.array import Array, atop, from_array

    if np.isnan(x.chunks[axis]).any():
        raise NotImplementedError("Slicing an array with unknown chunks with a "
                                  "dask.array of ints is not supported")

    # Calculate the offset at which each chunk starts along axis
    # e.g. chunks=(..., (5, 3, 4), ...) -> offset=[0, 5, 8]
    offset = np.roll(np.cumsum(x.chunks[axis]), 1)
    offset[0] = 0
    offset = from_array(offset, chunks=1)
    # Tamper with the declared chunks of offset to make atop align it with x[axis]
    offset = Array(offset.dask, name=offset.name, chunks=(x.chunks[axis], ), dtype=int)

    # Define axis labels for atop
    x_axes = 'abcdefghijklmnopqrstuvwxy'[:x.ndim]
    idx_axes = 'z'
    offset_axes = x_axes[axis]
    p_axes = x_axes[:axis + 1] + idx_axes + x_axes[axis + 1:]
    y_axes = x_axes[:axis] + idx_axes + x_axes[axis + 1:]

    # Calculate the cartesian product of every chunk of x vs. every chunk of idx
    p = atop(slice_with_int_dask_array_chunk,
             p_axes, x, x_axes, idx, idx_axes, offset, offset_axes,
             axis=axis, dtype=x.dtype)

    # Aggregate on the chunks of x along axis
    y = atop(slice_with_int_dask_array_aggregate,
             y_axes, idx, idx_axes, p, p_axes,
             concatenate=True, x_chunks=x.chunks[axis], axis=axis, dtype=x.dtype)
    return y


def slice_with_int_dask_array_chunk(x, idx, offset, axis):
    """Chunk kernel of `slice_with_int_dask_array_on_axis`.
    Slice 1 chunk of x by 1 chunk of idx.

    Returns ``x`` sliced along ``axis``, using only the elements of
    ``idx`` that fall inside the current chunk.
    """
    idx = idx - offset[0]
    idx_filter = (idx >= 0) & (idx < x.shape[axis])
    idx = idx[idx_filter]
    return x[[
        idx if i == axis else slice(None)
        for i in range(x.ndim)
    ]]


def slice_with_int_dask_array_aggregate(idx, chunk_outputs, x_chunks, axis):
    """Final aggregation kernel of `slice_with_int_dask_array_on_axis`.
    Aggregate all chunks of x by 1 chunk of idx.
    """
    x_chunk_offset = 0
    chunk_output_offset = 0

    # Assemble the final index that picks from the output of the previous
    # kernel by adding together one layer per chunk of x
    idx_final = np.zeros_like(idx)
    for x_chunk in x_chunks:
        idx_filter = (idx >= x_chunk_offset) & (idx < x_chunk_offset + x_chunk)
        idx_cum = np.cumsum(idx_filter)
        idx_final += np.where(idx_filter, idx_cum - 1 + chunk_output_offset, 0)
        x_chunk_offset += x_chunk
        if idx_cum.size > 0:
            chunk_output_offset += idx_cum[-1]

    return chunk_outputs[[
        idx_final if i == axis else slice(None)
        for i in range(chunk_outputs.ndim)
    ]]
