from .recursive_diff import recursive_diff


def recursive_eq(lhs, rhs, rel_tol=1e-09, abs_tol=0.0):
    """Wrapper around :func:`~xarray_extras..recursive_diff.recursive_diff`.
    Print out all differences and finally assert that there are none.
    """
    i = -1
    for i, diff in recursive_diff(lhs, rhs, rel_tol=rel_tol, abs_tol=abs_tol):
        print(diff)
    assert i == -1, "%d differences found" % (i + 1)
