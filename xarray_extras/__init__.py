import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("xarray_extras").version
except Exception:  # pragma: nocover
    # Local copy, not installed with setuptools
    __version__ = "999"

__all__ = ("__version__",)
