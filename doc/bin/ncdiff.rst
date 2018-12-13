ncdiff
======
Compare either two NetCDF files or all NetCDF files in two directories.

Usage
-----
::

    usage: ncdiff.py [-h]
                     [--engine {netcdf4,scipy,pydap,h5netcdf,pynio,cfgrib,pseudonetcdf}]
                     [--quiet] [--recursive] [--match MATCH] [--rtol RTOL]
                     [--atol ATOL] [--brief_dims DIM [DIM ...] | --brief]
                     lhs rhs

    Compare either two NetCDF files or all NetCDF files in two directories.

    positional arguments:
      lhs                   Left-hand-side NetCDF file or (if --recursive) directory
      rhs                   Right-hand-side NetCDF file or (if --recursive) directory

    optional arguments:
      -h, --help            show this help message and exit
      --engine {netcdf4,scipy,pydap,h5netcdf,pynio,cfgrib,pseudonetcdf},
      -e {netcdf4,scipy,pydap,h5netcdf,pynio,cfgrib,pseudonetcdf}
                            NeCDF engine (may require additional modules
      --quiet, -q           Suppress logging
      --recursive, -r       Compare all NetCDF files with matching names in two directories
      --match MATCH, -m MATCH
                            Bash wildcard match for file names when using --recursive (default: **/*.nc)
      --rtol RTOL           Relative comparison tolerance (default: 1e-9)
      --atol ATOL           Absolute comparison tolerance (default: 0)
      --brief_dims DIM [DIM ...]
                            Just count differences along one or more dimensions instead of printing them out individually
      --brief, -b           Just count differences for every variable instead of printing them out individually

    Examples:

    Compare two NetCDF files:
      ncdiff a.nc b.nc
    Compare all NetCDF files with identical names in two directories:
      ncdiff -r dir1 dir2


Chunking and RAM design
-----------------------
This tool does not support chunked files, or loading only part of
large datasets into memory at once. Instead, chunked datasets are
loaded as individual files. One variable at a time is then loaded
into memory completely, compared, and then discarded.

This has the big advantage of simplicity, but a few disadvantages:

- No option to compare datasets with mismatched prefixes
  (e.g. :file:`foo.*.nc` vs. :file:`bar.*.nc`).
- No option to compare chunked datasets that differ only in chunking
- Slower, as there is no option to skip loading over and over again
  variables that don't sit on the concat_dim.
  See also `xarray#2039 <https://github.com/pydata/xarray/issues/2039>`_.
- Huge RAM usage in case of monolithic variables

Further limitations
-------------------
- Won't compare NetCDF settings, e.g. store version, compression,
  chunking, etc.
- Doesn't support indices with duplicate elements
