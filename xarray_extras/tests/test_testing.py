import io
import sys
from contextlib import contextmanager
import pytest
from xarray_extras.testing import recursive_eq


@contextmanager
def capture_stdout():
    sys.stdout, backup = io.StringIO(), sys.stdout
    try:
        yield sys.stdout
    finally:
        sys.stdout = backup


def test_recursive_eq_success():
    with capture_stdout() as out:
        recursive_eq(0, 0)
    assert out.getvalue() == ''


def test_recursive_eq_fail():
    # Test the actual log lines dumped out by recursive_eq
    with capture_stdout() as out, pytest.raises(AssertionError):
        recursive_eq(('foo', ('bar', 'baz')),
                     ('foo', ('bar', 'asd', 'lol')))
    assert out.getvalue().splitlines() == [
        "[1]: RHS has 1 more elements than LHS: ['lol']",
        "[1][1]: baz != asd"
    ]
