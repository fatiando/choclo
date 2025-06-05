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
from choclo.prism._kernels import _kernel_iij


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
        _kernel_iij(x, y, z, radius)

    def test_accuracy_for_small_x(self):
        r"""
        Test numerical instabilities on ``kernel_iij``.

        Let's evaluate the kernel_iij on a set of shifted coordinates:

        - ``x``: values around zero, small enough to trigger potential
          floating point errors with the other coordinates.
        - ``y``: equal to zero.
        - ``z1`` and ``z2``: constant negative values, significantly greater
          than ``x`` to trigger trigger floating point errors.
          Assume ``z1 < z2``, where ``z1`` is bottom and ``z2`` is top (in
          shifted coordinates).

        The numerical instabilities are very noticeable after computing the
        difference between ``kernel_iij`` and ``z2`` and ``z1``.

        For small values of ``x``, the difference between the kernel evaluated
        on ``z2`` and ``z1`` can be approximated by first order Taylor series
        expansion:

        .. math::

            \delta k(x) \simeq
                \frac{1}{2} \frac{z_1^2 - z_2^2}{z_1^2 z_2^2} x
        """
        bottom, top = -5.0, -2.5
        spacing = 1e-6
        xmax = 1e-3
        n = int(xmax / spacing) + 1
        x = np.linspace(-xmax, xmax, n)
        kernel_diff = evaluate_kernel_iij(x, 0, top) - evaluate_kernel_iij(x, 0, bottom)
        kernel_diff_approx = approximate_kernel_diff(x, bottom, top)
        # Check if the approximation is close enough
        atol = kernel_diff_approx.max() * 1e-5
        np.testing.assert_allclose(kernel_diff, kernel_diff_approx, atol=atol)


def approximate_kernel_diff(x, z1, z2):
    r"""
    First order Taylor series expansion of ``kernel_iij`` difference.

    Approximate the difference between ``kernel_iij(x, 0, z1)`` and
    ``kernel_iij(x, 0, z2)`` with a first order Taylor series expansion:

    .. math::

        \delta k(x) \simeq
            \frac{1}{2} \frac{z_1^2 - z_2^2}{z_1^2 z_2^2} x

    Valid for ``z1 < z2 < 0``, ``|x| << |z2|``, and assuming ``y=0``.

    """
    # Make sure the argument values are correct
    assert z1 < z2
    assert z2 < 0
    return 0.5 * (z1**2 - z2**2) / (z1**2 * z2**2) * x


def evaluate_kernel_iij(x: np.ndarray, y: float, z: float):
    """
    Evaluate _kernel_iij on several x values, keeping y and z constant.
    """
    result = np.empty_like(x, dtype=np.float64)
    for i in range(x.size):
        radius = np.sqrt(x[i] ** 2 + y**2 + z**2)
        result[i] = _kernel_iij(x[i], y, z, radius)
    return result
