.. currentmodule:: xarray_extras

What's New
==========

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xarray
    np.random.seed(123456)


.. _whats-new.0.4.0:

v0.4.0 (Unreleased)
-------------------
- Moved ``recursive_diff``, ``recursive_eq`` and ``ncdiff``
  to their own package `recursive_diff <http://recursive_diff.readthedocs.io>`_
- Fixed bug in :func:`~xarray_extras.stack.proper_unstack` where unstacking
  coords with dtype=datetime64 would convert them to integer
- Mandatory flake8 in CI


.. _whats-new.0.3.0:

v0.3.0 (2018-12-13)
-------------------

- Changed license to Apache 2.0
- Increased minimum versions: dask >= 0.19, pandas >= 0.21,
  xarray >= 0.10.1, pytest >= 3.6
- New function :func:`~xarray_extras.stack.proper_unstack`
- New functions ``recursive_diff`` and ``ecursive_eq``
- New command-line tool ``ncdiff``
- Blacklisted Python 3.7 conda-forge builds in CI tests


.. _whats-new.0.2.2:

v0.2.2 (2018-07-24)
-------------------

- Fixed segmentation faults in :func:`~xarray_extras.csv.to_csv`
- Added conda-forge travis build
- Blacklisted dask-0.18.2 because of regression in argtopk(split_every=2)


.. _whats-new.0.2.1:

v0.2.1 (2018-07-22)
-------------------

- Added parameter nogil=True to :func:`~xarray_extras.csv.to_csv`, which will
  switch to a  C-accelerated implementation instead of pandas to_csv (albeit
  with caveats). Fixed deadlock in to_csv as well as compatibility with dask
  distributed. Pandas code (when using nogil=False) is not wrapped by a
  subprocess anymore, which means it won't be able to use more than 1 CPU
  (but compression can run in pipeline).
  to_csv has lost the ability to write to a buffer - only file paths are
  supported now.
- AppVeyor integration


.. _whats-new.0.2.0:

v0.2.0 (2018-07-15)
-------------------

- New function :func:`xarray_extras.csv.to_csv`
- Speed up interpolation for k=2 and k=3
- CI: Rigorous tracking of minimum dependency versions
- CI: Explicit support for Python 3.7


.. _whats-new.0.1.0:

v0.1.0 (2018-05-19)
-------------------

Initial release.