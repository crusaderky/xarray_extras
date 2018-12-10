import os
import pytest
import xarray
from xarray_extras.bin.ncdiff import main


DIR1 = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'mtf1'))
DIR2 = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'mtf2'))


a = xarray.Dataset(
    data_vars={
        'd1': ('x', [1, 2]),
        'd2': (('x', 'y'), [[1.0, 1.1], [1.2, 1.3]]),
        'd3': ('y', [3.0, 4.0])
    },
    coords={
        'x': [10, 20],
        'x2': ('x', [100, 200]),
        'y': ['y1', 'y2'],
    },
    attrs={
        'a1': 1, 'a2': 2
    })


def assert_stdout(capsys, expect):
    actual = capsys.readouterr().out
    print('Expect:')
    print(expect)
    print('Actual:')
    print(actual)
    assert expect == actual
    # Discard the print output above
    capsys.readouterr()


@pytest.mark.parametrize('argv', [
    ['d1/a.nc', 'd1/b.nc'],
    ['-q', 'd1/a.nc', 'd1/b.nc'],
    ['-b', 'd1/a.nc', 'd1/b.nc'],
    ['-r', 'd1', 'd2'],
    ['-b', '-r', 'd1', 'd2'],
    ['-r', '-m', '*/a.nc', 'd1', 'd2'],
])
def test_identical(tmpdir, capsys, argv):
    os.chdir(str(tmpdir))
    os.mkdir('d1')
    os.mkdir('d2')
    a.to_netcdf('d1/a.nc')
    a.to_netcdf('d1/b.nc')
    a.to_netcdf('d2/a.nc')
    a.to_netcdf('d2/b.nc')

    exit_code = main(argv)
    assert exit_code == 0
    assert_stdout(capsys, 'Found 0 differences\n')


@pytest.mark.parametrize('argv,out', [
    ([],
     '[attrs]: Pair a3:4 is in RHS only\n'
     '[attrs][a1]: 1 != 3 (abs: 2.0e+00, rel: 2.0e+00)\n'
     '[coords][x2][x=10]: 100 != 110 (abs: 1.0e+01, rel: 1.0e-01)\n'
     '[data_vars][d1][x=10]: 1 != 11 (abs: 1.0e+01, rel: 1.0e+01)\n'
     '[data_vars][d3][y=y1]: 3.0 != 3.01 (abs: 1.0e-02, rel: 3.3e-03)\n'
     'Found 5 differences\n'),
    (['-b'],
     '[attrs]: Pair a3:4 is in RHS only\n'
     '[attrs][a1]: 1 != 3 (abs: 2.0e+00, rel: 2.0e+00)\n'
     '[coords][x2]: 1 differences\n'
     '[data_vars][d1]: 1 differences\n'
     '[data_vars][d3]: 1 differences\n'
     'Found 5 differences\n'),
    (['--brief_dims', 'x', '--'],
     '[attrs]: Pair a3:4 is in RHS only\n'
     '[attrs][a1]: 1 != 3 (abs: 2.0e+00, rel: 2.0e+00)\n'
     '[coords][x2]: 1 differences\n'
     '[data_vars][d1]: 1 differences\n'
     '[data_vars][d3][y=y1]: 3.0 != 3.01 (abs: 1.0e-02, rel: 3.3e-03)\n'
     'Found 5 differences\n'),
    (['--atol', '5'],
     '[attrs]: Pair a3:4 is in RHS only\n'
     '[coords][x2][x=10]: 100 != 110 (abs: 1.0e+01, rel: 1.0e-01)\n'
     '[data_vars][d1][x=10]: 1 != 11 (abs: 1.0e+01, rel: 1.0e+01)\n'
     'Found 3 differences\n'),
    (['--rtol', '1e-1'],
     '[attrs]: Pair a3:4 is in RHS only\n'
     '[attrs][a1]: 1 != 3 (abs: 2.0e+00, rel: 2.0e+00)\n'
     '[data_vars][d1][x=10]: 1 != 11 (abs: 1.0e+01, rel: 1.0e+01)\n'
     'Found 3 differences\n'),
])
def test_singlefile(tmpdir, capsys, argv, out):
    b = a.copy(deep=True)
    b.d1[0] += 10
    b.d3[0] += .01
    b.attrs['a1'] = 3
    b.attrs['a3'] = 4
    b.x2[0] += 10
    a.to_netcdf('%s/a.nc' % tmpdir)
    b.to_netcdf('%s/b.nc' % tmpdir)

    exit_code = main(argv + ['%s/a.nc' % tmpdir, '%s/b.nc' % tmpdir])
    assert exit_code == 1
    assert_stdout(capsys, out)


# TODO: --engine
# TODO: --recursive
# TODO: --match
