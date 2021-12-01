#!/usr/bin/env python
from setuptools import setup

# Use hardcoded version when .git has been removed and this is not a package created by
# sdist. This is the case e.g. of a remote deployment with PyCharm.
setup(
    use_scm_version={"fallback_version": "999"},
    ext_modules=[
        Extension(
            'xarray_extras.kernels.np_to_csv', ['xarray_extras/kernels/np_to_csv.c']
        )
    ],
)
