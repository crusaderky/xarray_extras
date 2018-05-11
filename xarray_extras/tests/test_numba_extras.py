import numpy
import pytest
from xarray_extras.numba_extras import guvectorize


DTYPES = [
    # uint needs to appear before signed int:
    # https://github.com/numba/numba/issues/2934
    'uint8', 'uint16', 'uint32', 'uint64',
    'int8', 'int16', 'int32', 'int64',
    'float32', 'float64',
    'complex64', 'complex128'
]


@guvectorize('{T}[:], {T}[:]', '()->()')
def dumb_copy(x, y):
    for i in range(x.size):
        y.flat[i] = x.flat[i]


@pytest.mark.parametrize('dtype', DTYPES)
def test_guvectorize(dtype):
    x = numpy.arange(3, dtype=dtype)
    y = dumb_copy(x)
    numpy.testing.assert_equal(x, y)
    assert x.dtype == y.dtype
