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


class TestKerneliij:
    """
    Test ``kernel_iij`` for numerical instabilities.

    The previous implementation of ``kernel_iij`` suffered from numerical
    instabilities when x_i and x_j are close to zero and x_k is negative:
    adding the radius with x_k might lead to high float precision errors.
    In the special case where radius is exactly equal to x_k (up to float point
    precision, but x_i or x_j are not zero, it will lead to a division by zero
    error.
    """

    def test_kernel_iij_division_by_zero(self):
        """
        Test if we don't get division by zero.

        Evaluate ``kernel_iij`` on ``x > 0``, ``y=0``, ``z < 0`` and
        ``x << |z|``. If we use the previous implementation of kernel_iij, this
        test should fail.
        """
        x, y, z = 1e-12, 0, -500_000
        radius = np.sqrt(x**2 + y**2 + z**2)
        kernel_een(x, y, z, radius)

    def test_instabilities(self):
        """
        Test numerical instabilities on ``kernel_iij``.

        Let's evaluate the kernel_iij on a set of shifted coordinates:

        - ``x``: values around zero, small enough to trigger potential
          floating point errors with the other coordinates.
        - ``y``: equal to zero.
        - ``z``: constant negative value, significantly greater than ``x`` to
          trigger trigger floating point errors.

        Computing the difference between
        The kernel_iij for these points should behave as a monotonic decreasing
        function with x.
        """
        delta =
        x = np.linspace(-1e4, 1e4)
