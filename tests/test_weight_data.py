import numpy as np
import pytest

from pycalmet.weight_data import read_weight_data


def _weight_text():
    return """header one
header two
header three
               0.0 0.0 2 2 1.0
               0.0 0.0 1 1 2.0


            10.0000
      1 2
      0.10 0.20
      0.30 0.40
      end
            30.0000
      1 2
      0.50 0.60
      0.70 0.80
      end
"""


def test_weight_data_reverses_north_to_south_rows(tmp_path):
    path=tmp_path/'wt.dat'; path.write_text(_weight_text(),encoding='ascii')
    got=read_weight_data(path,2,2,[10.,30.])
    np.testing.assert_allclose(got[0],[[.3,.4],[.1,.2]])
    np.testing.assert_allclose(got[1],[[.7,.8],[.5,.6]])


def test_weight_data_checks_levels(tmp_path):
    path=tmp_path/'wt.dat'; path.write_text(_weight_text(),encoding='ascii')
    with pytest.raises(ValueError,match='levels'): read_weight_data(path,2,2,[11.,30.])
