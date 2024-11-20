# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Additional tests to prism kernels.
"""

import numpy as np
import pytest

from choclo.prism import (
    kernel_ee,
    kernel_eee,
    kernel_een,
    kernel_eeu,
    kernel_en,
    kernel_enn,
    kernel_enu,
    kernel_eu,
    kernel_euu,
    kernel_nn,
    kernel_nnn,
    kernel_nnu,
    kernel_nu,
    kernel_nuu,
    kernel_uu,
    kernel_uuu,
)


class TestThirdOrderKernelsOnVertex:
    """
    Test if third-order kernels evaluated on a vertex return np.nan.

    This functionality is meant to be supported for the public usage of this
    kernels, although the forward modelling functions within Choclo won't use
    it since they filter out singular points before they call kernels.
    """

    SECOND_ORDER_KERNELS = (
        kernel_ee,
        kernel_nn,
        kernel_uu,
        kernel_en,
        kernel_eu,
        kernel_nu,
    )
    THIRD_ORDER_KERNELS = (
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

    @pytest.mark.parametrize("kernel", THIRD_ORDER_KERNELS)
    def test_third_order_kernels_on_vertex(self, kernel):
        easting, northing, upward = 0.0, 0.0, 0.0
        radius = 0.0
        result = kernel(easting, northing, upward, radius)
        assert np.isnan(result)

    @pytest.mark.parametrize("kernel", SECOND_ORDER_KERNELS)
    def test_second_order_kernels_on_vertex(self, kernel):
        easting, northing, upward = 0.0, 0.0, 0.0
        radius = 0.0
        result = kernel(easting, northing, upward, radius)
        assert np.isnan(result)
