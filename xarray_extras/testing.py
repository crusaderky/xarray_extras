"""Tools for unit testing
"""
from .recursive_diff import recursive_diff


def recursive_eq(lhs, rhs, rel_tol=1e-09, abs_tol=0.0):
    """Wrapper around :func:`~xarray_extras..recursive_diff.recursive_diff`.
    Print out all differences and finally assert that there are none.
    """
    diffs_iter = recursive_diff(lhs, rhs, rel_tol=rel_tol, abs_tol=abs_tol)
    i = -1
    for i, diff in enumerate(diffs_iter):
        print(diff)
    i += 1
    assert i == 0, "%d differences found" % i
