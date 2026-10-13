import numpy as np
import pycalmet
from pycalmet.legacy._compat.calmet_numerics import _real4_libm


def test_legacy_api_is_separate_from_default_api():
    assert pycalmet.legacy.calwrf is not pycalmet.calwrf
    assert pycalmet.legacy.calmet is not pycalmet.calmet


def test_legacy_real4_intrinsic_preserves_single_precision_result():
    value = _real4_libm('sinf', np.float32(1.234567))
    assert isinstance(value, np.float32)
    assert value.view(np.uint32) == np.uint32(1064413783)
