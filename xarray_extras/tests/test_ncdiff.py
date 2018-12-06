import os.path
import pytest
from xarray_extras.bin.ncdiff import main


DIR1 = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'mtf1'))
DIR2 = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'mtf2'))


def assert_stdout(capsys, expect):
    actual = capsys.readouterr().out
    print("Expected:")
    print(expect)
    print("Got:")
    print(actual)
    assert set(expect.splitlines()) == set(actual.splitlines())


@pytest.mark.parametrize('argv', [
    ['-b', DIR1 + '/MyCube1.nc', DIR1 + '/MyCube1.nc'],
    [DIR1 + '/MyCube1.nc', DIR1 + '/MyCube1.nc'],
    ['-r', '-b', '-m', '*.nc', DIR1, DIR1],
    ['-r', '-m', '*.nc', DIR1, DIR1],
])
def test_identical(capsys, argv):
    exit_code = main(argv)
    assert exit_code == 0
    assert_stdout(capsys, 'Found 0 differences\n')


def test_brief(capsys):
    exit_code = main(['-b', DIR1 + '/MyCube1.nc', DIR1 + '/MyCube3.nc'])
    assert exit_code == 1
    assert_stdout(
        capsys,
        '[data_vars][FX]: 1 differences\n'
        '[index][instr_id]: 1 differences\n'
        '[index][scenario]: 2 differences\n'
        'Found 3 differences\n')


def test_brief_dims(capsys):
    exit_code = main(['--brief_dims', 'scenario', '--',
                      DIR1 + '/MyCube1.nc', DIR1 + '/MyCube3.nc'])
    assert exit_code == 1
    assert_stdout(
        capsys,
        '[data_vars][FX][fx_id=EUR, timestep=2012-12-31 00:00:00]: 1 differences\n'  # noqa
        '[index][instr_id]: instr_id=MyCFInstr2 is in LHS only\n'
        '[index][scenario]: 2 differences\n'
        'Found 3 differences\n')


def test_verbose(capsys):
    exit_code = main([DIR1 + '/MyCube1.nc', DIR1 + '/MyCube3.nc'])
    assert exit_code == 1
    assert_stdout(
        capsys,
        '[data_vars][FX][fx_id=EUR, scenario=scen2, timestep=2012-12-31 00:00:00]: 50.0 != 5.0 (abs: -4.5e+01, rel: -9.0e-01)\n'  # noqa
        '[index][instr_id]: instr_id=MyCFInstr2 is in LHS only\n'
        '[index][scenario]: scenario=scen1 is in LHS only\n'
        '[index][scenario]: scenario=scen3 is in RHS only\n'
        'Found 4 differences\n')


def test_recursive_brief(capsys):
    exit_code = main(['-r', '-b', DIR1, DIR2])
    assert exit_code == 1
    assert_stdout(
        capsys,
        'Pair MyCube2.nc:<xarray.Dataset> ... is in LHS only\n'
        'Pair MyCube3.nc:<xarray.Dataset> ... is in LHS only\n'
        'Pair Small_All_RiskDrivers_Shredded_MP.nc:<xarray.Dataset> ... is in LHS only\n'  # noqa
        'Pair Small_All_RiskDrivers_Shredded_Market.nc:<xarray.Dataset> ... is in LHS only\n'  # noqa
        'Pair Small_merged.nc:<xarray.Dataset> ... is in LHS only\n'
        '[MyCube1.nc][coords][currency]: 1 differences\n'
        '[MyCube1.nc][data_vars][instruments]: 1 differences\n'
        '[MyCube1.nc][index][instr_id]: 1 differences\n'
        'Found 8 differences\n')


def test_recursive_verbose(capsys):
    exit_code = main(['-r', '-m', DIR1, DIR2])
    assert exit_code == 1
    assert_stdout(
        capsys,
        'Pair MyCube2.nc:<xarray.Dataset> ... is in LHS only\n'
        'Pair MyCube3.nc:<xarray.Dataset> ... is in LHS only\n'
        'Pair Small_All_RiskDrivers_Shredded_MP.nc:<xarray.Dataset> ... is in LHS only\n'  # noqa
        'Pair Small_All_RiskDrivers_Shredded_Market.nc:<xarray.Dataset> ... is in LHS only\n'  # noqa
        'Pair Small_merged.nc:<xarray.Dataset> ... is in LHS only\n'
        '[MyCube1.nc][coords][currency][instr_id=MyCFInstr2]: EUR != GBP\n'
        '[MyCube1.nc][data_vars][instruments][attribute=THEO/Value, instr_id=MyCFInstr1, scenario=scen1, timestep=2012-12-31 00:00:00]: 2.0 != 7.0 (abs: 5.0e+00, rel: 2.5e+00)\n'  # noqa
        '[MyCube1.nc][index][instr_id]: instr_id=MyCFInstr3 is in LHS only\n'
        'Found 8 differences\n')


# TODO: test --match
# TODO: test --engine
