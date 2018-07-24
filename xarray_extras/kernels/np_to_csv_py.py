"""Thin ctypes wrapper around :file:`np_to_csv.c`.
This is a helper module of :mod:`xarray_extras.kernels.csv`.
"""
import ctypes
import numpy as np
from . import np_to_csv
np_to_csv = np.ctypeslib.load_library('np_to_csv', np_to_csv.__file__)


np_to_csv.snprintcsvd.argtypes = [
    ctypes.c_char_p,  # char * buf
    ctypes.c_int32,   # int bufsize
    # const double * array
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ctypes.c_int32,   # int h
    ctypes.c_int32,   # int w
    ctypes.c_char_p,  # const char * index
    ctypes.c_char_p,  # const char * fmt
    ctypes.c_bool,    # bool trim_zeros
    ctypes.c_char_p,  # const char * na_rep
]
np_to_csv.snprintcsvd.restype = ctypes.c_int32


np_to_csv.snprintcsvi.argtypes = [
    ctypes.c_char_p,  # char * buf
    ctypes.c_int32,   # int bufsize
    # const int64_t * array
    np.ctypeslib.ndpointer(dtype=np.int64, ndim=2, flags='C_CONTIGUOUS'),
    ctypes.c_int32,   # int h
    ctypes.c_int32,   # int w
    ctypes.c_char_p,  # const char * index
    ctypes.c_char,    # char sep
]
np_to_csv.snprintcsvi.restype = ctypes.c_int32


def snprintcsvd(a, index, sep=',', fmt=None, na_rep=''):
    """Convert array to CSV.

    :param a:
        1D or 2D numpy array of floats
    :param str index:
        newline-separated list of prefixes for every row of a
    :param str sep:
        cell separator
    :param str fmt:
        printf formatting string for a single float number
        Set to None to replicate pandas to_csv default behaviour
    :param str na_rep:
        string representation of NaN
    :return:
        CSV file contents, binary-encoded in ascii format.
        The line terminator is always \n on all OSs.
    """
    if a.ndim == 1:
        a = a.reshape((-1, 1))
    if a.ndim != 2 or a.dtype.kind != 'f':
        raise ValueError("Expected 2d numpy array of floats")
    a = a.astype(np.float64)
    a = np.ascontiguousarray(a)
    if len(sep) != 1:
        raise ValueError("sep must be exactly 1 character")
    bsep = sep.encode('ascii')

    # Test fmt while in Python - much better to get
    # an Exception here than a segfault in C!
    if fmt is not None:
        fmt % 1.23  # noqa
        bfmt = fmt.encode('ascii') + bsep
        trim_zeros = False
    else:
        bfmt = b'%f' + bsep
        trim_zeros = True
    bna_rep = na_rep.encode('ascii') + bsep
    # We're relying on the fact that ascii is a strict subset of UTF-8
    bindex = index.encode('utf-8')

    # Blindly try ever-larger bufsizes until it fits
    # The first iteration should be sufficient in all but the most
    # degenerate cases.
    # FIXME: is there a better way?
    cellsize = 40
    while True:
        bufsize = cellsize * a.size + len(bindex)
        buf = ctypes.create_string_buffer(bufsize)
        nchar = np_to_csv.snprintcsvd(buf, bufsize, a, a.shape[0], a.shape[1],
                                      bindex, bfmt, trim_zeros, bna_rep)
        if nchar < bufsize:
            return buf[:nchar]
        cellsize *= 2


def snprintcsvi(a, index, sep=','):
    """Convert array to CSV.

    :param a:
        1D or 2D numpy array of integers
    :param str index:
        newline-separated list of prefixes for every row of a
    :param str sep:
        cell separator
    :return:
        CSV file contents, binary-encoded in ascii format.
        The line terminator is always \n on all OSs.
    """
    if a.ndim == 1:
        a = a.reshape((-1, 1))
    if a.ndim != 2 or a.dtype.kind != 'i':
        raise ValueError("Expected 2d numpy array of ints")
    a = a.astype(np.int64)
    a = np.ascontiguousarray(a)
    if len(sep) != 1:
        raise ValueError("sep must be exactly 1 character")
    bsep = sep.encode('ascii')
    # We're relying on the fact that ascii is a strict subset of UTF-8
    bindex = index.encode('utf-8')

    cellsize = 22  # len('%d' % -2**64) + 1
    bufsize = cellsize * a.size + len(bindex)
    buf = ctypes.create_string_buffer(bufsize)
    nchar = np_to_csv.snprintcsvi(
        buf, bufsize, a, a.shape[0], a.shape[1], bindex, bsep)
    assert nchar < bufsize
    return buf[:nchar]
