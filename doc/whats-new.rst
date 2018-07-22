.. currentmodule:: xarray_extras

What's New
==========

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xarray
    np.random.seed(123456)

.. _whats-new.0.2.1:

v0.2.1 (2018-07-22)
-------------------

- Added parameter nogil=True to :func:`xarray_extras.csv.to_csv`, which will
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