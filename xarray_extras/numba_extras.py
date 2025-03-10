"""Extensions to numba"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numba

_DTYPES = (
    # uint needs to appear before signed int:
    # https://github.com/numba/numba/issues/2934
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "complex64",
    "complex128",
)


def guvectorize(
    signature: str, layout: str, **kwargs: Any
) -> Callable[[Callable], Any]:
    """Convenience wrapper around :func:`numba.guvectorize`.
    Generate signature for all possible data types and set a few healthy
    defaults.

    :param str signature:
        numba signature, containing {T}
    :param str layout:
        as in :func:`numba.guvectorize`
    :param kwargs:
        passed verbatim to :func:`numba.guvectorize`.
        This function changes the default for cache from False to True.

    example::

        guvectorize("{T}[:], {T}[:]", "(i)->(i)")

    Is the same as::

        numba.guvectorize([
            "float32[:], float32[:]",
            "float64[:], float64[:]",
            ...
        ], "(i)->(i)", cache=True)

    .. note::
       Discussing upstream fix; see
       `<https://github.com/numba/numba/issues/2936>`_.
    """
    if "{T}" in signature:
        signatures = [signature.format(T=dtype) for dtype in _DTYPES]
    else:
        signatures = [signature]
    kwargs.setdefault("cache", True)
    return numba.guvectorize(signatures, layout, **kwargs)
