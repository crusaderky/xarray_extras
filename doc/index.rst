xarray_extras: Advanced algorithms for xarray
=============================================
This module offers several extensions to :mod:`xarray`, which could not be
included into the main xarray module because they fall into one or more of the
following categories:

- They're too experimental
- They're too niche
- They introduce major new dependencies (e.g. :mod:`numba` or a C compiler)
- They would be better done by doing major rework on multiple packages, and
  then one would need to wait for said changes to reach a stable release of
  each package - *in the right order*.

The API of xarray-extras is unstable by definition, as features will be
progressively migrated upwards towards xarray, dask, numpy etc.

Features
--------
:doc:`api/csv`
    Multi-threaded CSV writer, much faster than
    :meth:`pandas.DataFrame.to_csv`, with full support for :mod:`dask` and
    :mod:`distributed`
:doc:`api/cumulatives`
    Advanced cumulative sum/productory/mean functions
:doc:`api/interpolate`
    dask-optimized n-dimensional spline interpolation
:doc:`api/numba_extras`
    Additions to :mod:`numba`
:doc:`api/sort`
    Advanced sort/take functions

Index
-----

.. toctree::

   installing
   whats-new

API Reference
-------------

.. toctree::

   api/csv
   api/cumulatives
   api/interpolate
   api/numba_extras
   api/sort

License
-------

xarray-extras is available under the open source `Apache License`__.

__ http://www.apache.org/licenses/LICENSE-2.0.html