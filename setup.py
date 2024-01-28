from setuptools import Extension, setup

setup(
    use_scm_version=True,
    # Compile CPython extensions
    ext_modules=[
        Extension(
            "xarray_extras.kernels.np_to_csv", ["xarray_extras/kernels/np_to_csv.c"]
        )
    ],
)
