"""
Additional tests to prism kernels.
"""

import pytest
import numpy as np

from choclo.prism import (
    kernel_eee,
    kernel_nnn,
    kernel_uuu,
    kernel_een,
    kernel_eeu,
    kernel_enn,
    kernel_nnu,
    kernel_euu,
    kernel_nuu,
    kernel_enu,
)


class TestThirdOrderKernelsOnVertex:
    """
    Test if third-order kernels evaluated on a vertex return np.nan.

    This functionality is meant to be supported for the public usage of this
    kernels, although the forward modelling functions within Choclo won't use
    it since they filter out singular points before they call kernels.
    """

    KERNELS = (
        kernel_eee,
        kernel_nnn,
        kernel_uuu,
        kernel_een,
        kernel_eeu,
        kernel_enn,
        kernel_nnu,
        kernel_euu,
        kernel_nuu,
        kernel_enu,
    )

    @pytest.mark.parametrize("kernel", KERNELS)
    def test_third_order_kernels_on_vertex(self, kernel):
        easting, northing, upward = 0.0, 0.0, 0.0
        radius = 0.0
        result = kernel(easting, northing, upward, radius)
        assert np.isnan(result)
