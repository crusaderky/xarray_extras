import importlib
from distutils.version import LooseVersion
import pytest


def _import_or_skip(modname, minversion=None):
    """Build skip markers for a optional module

    :param str modname:
        Name of the optional module
    :param str minversion:
        Minimum required version
    :return:
        Tuple of

        has_module (bool)
            True if the module is available and >= minversion
        requires_module (decorator)
            Tests decorated with it will only run if the module is available
            and >= minversion
    """
    reason = 'requires %s' % modname
    if minversion:
        reason += '>=%s' % minversion

    try:
        mod = importlib.import_module(modname)
        has = True
    except ImportError:
        has = False
    if (has and minversion
            and LooseVersion(mod.__version__) < LooseVersion(minversion)):
        has = False

    func = pytest.mark.skipif(not has, reason=reason)
    return has, func


has_h5netcdf, requires_h5netcdf = _import_or_skip('h5netcdf')
