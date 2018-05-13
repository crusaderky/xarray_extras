sort
====

.. automodule:: xarray_extras.sort
    :members:
    :undoc-members:
    :show-inheritance:


An example that uses all of the above functions is *source attribution*.
Given a generic function :math:`y = f(x_{0}, x_{1}, ..., x_{i})`, which is
embarassingly parallel along a given dimension, one wants to find:

- the top k elements of y along the dimension
- the elements of all x's that generated the top k elements of y

.. code::

    >>> from xarray import DataArray
    >>> from xarray_extras.sort import *
    >>> x = DataArray([[5, 3, 2, 8, 1],
    >>>                [0, 7, 1, 3, 2]], dims=['x', 's'])
    >>> y = x.sum('x')  # y = f(x), embarassingly parallel among dimension 's'
    >>> y
    <xarray.DataArray (s: 5)>
    array([ 5, 10,  3, 11,  3])
    Dimensions without coordinates: s
    >>> top_y = topk(y, 3, 's')
    >>> top_y
    <xarray.DataArray (s: 3)>
    array([11, 10,  5])
    Dimensions without coordinates: s
    >>> top_x = take_along_dim(x, argtopk(y, 3, 's'), 's')
    >>> top_x
    <xarray.DataArray (x: 2, s: 3)>
    array([[8, 3, 5],
           [3, 7, 0]])
    Dimensions without coordinates: x, s
