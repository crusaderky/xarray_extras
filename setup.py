#!/usr/bin/env python
import os
import re
import warnings

from setuptools import find_packages, setup, Extension


MAJOR = 0
MINOR = 3
MICRO = 0
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''


DISTNAME = 'xarray_extras'
LICENSE = 'LGPL'
AUTHOR = 'Guido Imperiale'
AUTHOR_EMAIL = 'guido.imperiale@gmail.com'
URL = 'https://github.com/crusaderky/xarray_extras'
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: GNU Lesser General Public License v3 or later '
    '(LGPLv3+)',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering',
]

INSTALL_REQUIRES = [
    'dask >= 0.19',
    'numba >= 0.34',
    'numpy >= 1.13',
    'pandas >= 0.21',
    'scipy >= 1.0',
    'xarray >= 0.10.1',
]
TESTS_REQUIRE = ['pytest >= 3.6']

DESCRIPTION = "Advanced / experimental algorithms for xarray"
LONG_DESCRIPTION = """

"""  # noqa

# Code to extract and write the version copied from pandas.
# Used under the terms of pandas's license, see licenses/PANDAS_LICENSE.
FULLVERSION = VERSION
write_version = True

if not ISRELEASED:
    import subprocess
    FULLVERSION += '.dev'

    pipe = None
    for cmd in ['git', 'git.cmd']:
        try:
            pipe = subprocess.Popen(
                [cmd, "describe", "--always", "--match", "v[0-9]*"],
                stdout=subprocess.PIPE)
            (so, serr) = pipe.communicate()
            if pipe.returncode == 0:
                break
        except BaseException:
            pass

    if pipe is None or pipe.returncode != 0:
        # no git, or not in git dir
        if os.path.exists('xarray_extras/version.py'):
            warnings.warn(
                "WARNING: Couldn't get git revision,"
                " using existing xarray_extras/version.py")
            write_version = False
        else:
            warnings.warn(
                "WARNING: Couldn't get git revision,"
                " using generic version string")
    else:
        # have git, in git dir, but may have used a shallow clone (travis does
        # this)
        rev = so.strip()
        rev = rev.decode('ascii')

        if not rev.startswith('v') and re.match("[a-zA-Z0-9]{7,9}", rev):
            # partial clone, manually construct version string
            # this is the format before we started using git-describe
            # to get an ordering on dev version strings.
            rev = "v%s+dev.%s" % (VERSION, rev)

        # Strip leading v from tags format "vx.y.z" to get th version string
        FULLVERSION = rev.lstrip('v')

        # make sure we respect PEP 440
        FULLVERSION = FULLVERSION.replace("-", "+dev", 1).replace("-", ".")

else:
    FULLVERSION += QUALIFIER


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), 'xarray_extras', 'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()


if write_version:
    write_version_py()

setup(name=DISTNAME,
      version=FULLVERSION,
      license=LICENSE,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      classifiers=CLASSIFIERS,
      description=DESCRIPTION,
      entry_points={'console_scripts': [
          'ncdiff = xarray_extras.bin.ncdiff:main',
      ]},
      ext_modules=[Extension('xarray_extras.kernels.np_to_csv',
                             ['xarray_extras/kernels/np_to_csv.c'])],
      long_description=LONG_DESCRIPTION,
      install_requires=INSTALL_REQUIRES,
      tests_require=TESTS_REQUIRE,
      url=URL,
      packages=find_packages(),
      package_data={'xarray_extras': ['tests/data/*']})
