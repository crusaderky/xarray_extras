#!/usr/bin/env python
"""Numba kernels for :mod:`cumulatives`
"""
from ..numba_extras import guvectorize


@guvectorize("{T}[:], intp[:], {T}[:]", "(i),(j)->()")
def compound_sum(x, c, y):
    """y = x[c[0]] + x[c[1]] + ... x[c[n]]
    until c[i] != -1
    """
    acc = 0
    for i in c:
        if i == -1:
            break
        acc += x[i]
    y[0] = acc


@guvectorize("{T}[:], intp[:], {T}[:]", "(i),(j)->()")
def compound_prod(x, c, y):
    """y = x[c[0]] * x[c[1]] * ... x[c[n]]
    until c[i] != -1
    """
    acc = 1
    for i in c:
        if i == -1:
            break
        acc *= x[i]
    y[0] = acc


@guvectorize("{T}[:], intp[:], {T}[:]", "(i),(j)->()")
def compound_mean(x, c, y):
    """y = mean(x[c[0]], x[c[1]], ... x[c[n]])
    until c[i] != -1
    """
    acc = 0
    j = 0  # Initialise j explicitly for when x.shape == (0, )
    for j, i in enumerate(c):
        if i == -1:
            break
        acc += x[i]
    else:
        # Reached the end of the row
        j += 1
    y[0] = acc / j
