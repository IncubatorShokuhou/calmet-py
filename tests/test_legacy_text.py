import pytest

from pycalmet.legacy_text import gfortran_real_pair


@pytest.mark.parametrize(('values','expected'),(
    ((0.,0.),'   0.00000000       0.00000000    \n'),
    ((12.5,-8.25),'   12.5000000      -8.25000000    \n'),
    ((.001234567,1234567.),'   1.23456703E-03   1234567.00    \n'),
    ((-0.,0.),'  -0.00000000       0.00000000    \n'),
    ((-.01,-1.e9),'  -9.99999978E-03  -1.00000000E+09\n'),
))
def test_gfortran_list_directed_real_pair(values,expected):
    assert gfortran_real_pair(*values)==expected
