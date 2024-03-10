from __future__ import annotations

try:
    from xarray.namedarray.pycompat import array_type
except ImportError:  # <2024.2.0
    from xarray.core.pycompat import array_type  # type: ignore[no-redef]

dask_array_type = array_type("dask")
