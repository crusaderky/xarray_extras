#!/usr/bin/env python
"""Compare either two NetCDF files or all NetCDF files in two directories.

See :doc:`bin/ncdiff`
"""
import argparse
import glob
import logging
import os
import sys
import xarray
from ..recursive_diff import recursive_diff


LOGFORMAT = '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s'


def argparser():
    """Return precompiled ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Compare either two NetCDF files or all NetCDF files in "
                    "two directories.",
        epilog="Examples:\n\n"
               "Compare two NetCDF files:\n"
               "  ncdiff a.nc b.nc\n"
               "Compare all NetCDF files with identical names in two "
               "directories:\n"
               "  ncdiff -r dir1 dir2\n",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '--engine', '-e',
        help='NeCDF engine (may require additional modules',
        choices=['netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio', 'cfgrib',
                 'pseudonetcdf'])
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress logging')

    parser.add_argument(
        '--recursive', '-r', action='store_true',
        help='Compare all NetCDF files with matching names in two directories')
    parser.add_argument(
        '--match', '-m', default='**/*.nc',
        help="Bash wildcard match for file names when using --recursive "
             "(default: **/*.nc)")

    parser.add_argument(
        '--rtol', type=float, default=1e-9,
        help="Relative comparison tolerance (default: 1e-9)")
    parser.add_argument(
        '--atol', type=float, default=0,
        help="Absolute comparison tolerance (default: 0)")

    brief = parser.add_mutually_exclusive_group()
    brief.add_argument(
        '--brief_dims', nargs='+', default=(), metavar='DIM',
        help="Just count differences along one or more dimensions instead of "
             "printing them out individually")
    brief.add_argument(
        '--brief', '-b', action='store_true',
        help="Just count differences for every variable instead of printing "
             "them out individually")

    parser.add_argument(
        'lhs',
        help="Left-hand-side NetCDF file or (if --recursive) directory")
    parser.add_argument(
        'rhs',
        help="Right-hand-side NetCDF file or (if --recursive) directory")

    return parser


def open_netcdf(fname, engine=None):
    """Open a single NetCDF dataset
    Read the metadata into RAM. Do not load the actual data.

    :param str fname:
        path to .nc file
    :param str engine:
        NetCDF engine (see :func:`xarray.open_dataset`)
    :returns:
        :class:`xarray.Dataset`
    """
    # At the moment of writing, h5netcdf is the only engine
    # supporting LZF compression
    logging.info("Opening %s", fname)
    return xarray.open_dataset(fname, engine=engine, chunks={})


def recursive_open_netcdf(path, match, engine=None):
    """Recursively find and open all NetCDF files that exist in any of
    the given paths.

    :param str path:
        Root directory to search into
    :param str engine:
        NetCDF engine (see :func:`xarray.open_dataset`)
    :returns:
        dict of {relative file name: dataset}
    """
    cwd = os.getcwd()
    os.chdir(path)
    try:
        fnames = glob.glob(match, recursive=True)
    finally:
        os.chdir(cwd)

    # We don't invoke open_netcdf() directly inside the pushd context
    # to get a prettier logging message on the file being opened
    logging.info("Opening %d NetCDF stores from %s", len(fnames), path)
    return {fname: open_netcdf(os.path.join(path, fname), engine=engine)
            for fname in fnames}


def main(argv=None):
    """Parse command-line arguments, load all files, and invoke recursive_diff

    :returns:
        exit code
    """
    # Parse command-line arguments and init logging
    args = argparser().parse_args(argv)
    if args.brief:
        args.brief_dims = 'all'

    if args.quiet:
        loglevel = logging.WARNING
    else:
        loglevel = logging.INFO

    # Don't init logging when running inside unit tests
    if argv is None:
        logging.basicConfig(level=loglevel, format=LOGFORMAT)

    # Load metadata of all NetCDF stores
    # Leave actual data on disk
    if args.recursive:
        lhs = recursive_open_netcdf(args.lhs, args.match, engine=args.engine)
        rhs = recursive_open_netcdf(args.rhs, args.match, engine=args.engine)
    else:
        lhs = open_netcdf(args.lhs, engine=args.engine)
        rhs = open_netcdf(args.rhs, engine=args.engine)

    logging.info("Comparing...")
    # 1. Load a pair of NetCDF variables fully into RAM
    # 2. compare them
    # 3. print all differences
    # 4. free the RAM
    # 5. proceed to next pair
    diff_iter = recursive_diff(
        lhs, rhs,
        abs_tol=args.atol, rel_tol=args.rtol,
        brief_dims=args.brief_dims)

    diff_count = 0
    for diff in diff_iter:
        diff_count += 1
        print(diff)

    print("Found %d differences" % diff_count)
    if diff_count:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
