.. _installing:

Installation
============

Required dependencies
---------------------

- Python 3.5 or later
- `scipy <https://docs.scipy.org/doc/>`__
- `xarray <http://xarray.pydata.org/>`__
- `dask <http://dask.pydata.org>`__
- `numba <http://numba.pydata.org>`__
- C compiler (only if building from sources)

Deployment
----------

- With pip: :command:`pip install xarray-extras`
- With `anaconda <https://www.anaconda.com/>`_:
  :command:`conda install -c conda-forge xarray-extras`

Testing
-------

To run the test suite after installing xarray_extras, first install (via pypi or conda)

- `py.test <https://pytest.org>`__: Simple unit testing library

and run
``py.test --pyargs xarray_extras``.

