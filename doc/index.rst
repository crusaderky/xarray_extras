xarray_extras: Advanced algorithms for xarray
=============================================
This module offers several extensions to `xarray <http://xarray.pydata.org/>`_,
which could not be included into the main module because they fall into one or
more of the following categories:

- They're too experimental
- They're too niche
- They introduce major new dependencies (e.g.
  `numba <http://numba.pydata.org/>`_ or a C compiler)
- They would be better done by doing major rework on multiple packages, and
  then one would need to wait for said changes to reach a stable release of
  each package - *in the right order*.

The API of xarray-extras is unstable by definition, as features will be
progressively migrated upwards towards xarray, dask, numpy, pandas, etc.

Features
--------
:doc:`api/csv`
    Multi-threaded CSV writer, much faster than
    :meth:`pandas.DataFrame.to_csv`, with full support for
    `dask <http://dask.org/>`_ and
    `dask distributed <http://distributed.dask.org/>`_.
:doc:`api/cumulatives`
    Advanced cumulative sum/productory/mean functions
:doc:`api/interpolate`
    dask-optimized n-dimensional spline interpolation
:doc:`api/numba_extras`
    Additions to `numba <http://numba.pydata.org/>`_
:doc:`api/sort`
    Advanced sort/take functions
:doc:`api/stack`
    Tools for stacking/unstacking dimensions


Index
-----

.. toctree::

   installing
   whats-new
   api/csv
   api/cumulatives
   api/interpolate
   api/numba_extras
   api/sort
   api/stack


Credits
-------
- :func:`~xarray_extras.stack.proper_unstack` was originally developed by
  Legal & General and released to the open source community in 2018.
- All boilerplate is from
  `python_project_template <https://github.com/crusaderky/python_project_template>`_,
  which in turn is from `xarray <http://xarray.pydata.org/>`_.

License
-------

xarray-extras is available under the open source `Apache License`__.

__ http://www.apache.org/licenses/LICENSE-2.0.html