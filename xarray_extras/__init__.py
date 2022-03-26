import importlib.metadata

try:
    __version__ = importlib.metadata.version("xarray_extras")
except importlib.metadata.PackageNotFoundError:  # pragma: nocover
    # Local copy, not installed with pip
    __version__ = "999"

__all__ = ("__version__",)
