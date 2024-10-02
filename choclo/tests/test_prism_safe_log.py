# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the _safe_log private function for prisms.
"""

import numpy as np
import numpy.testing as npt
import pytest

from ..prism._kernels import _safe_log


class TestSafeLogFixedX:
    r"""
    Test some properties of ``_safe_log`` when ``x`` is fixed.

    We define:

    .. math::

        f(x, y, z) = \ln(x + \sqrt{x^2 + y^2 + z^2})

    When fixing ``x``, this function has some continuity, symmetry and monotony
    properties.

    For simplicity, let's define:

    .. math::

        g(z) = f(x=x_0, y=0, z) = \ln(x_0 + \sqrt{x_0^2 + z^2})

    with :math:`x_0 = \text{constant}`.
    """

    def eval_safe_log(self, z, x_0=0.0):
        r = np.sqrt(x_0**2 + z**2)
        return _safe_log(x_0, 0.0, z, r)

    @pytest.mark.parametrize(
        "x_0", (-10.0, 0.0, 10.0), ids=("x_0=-10.0", "x_0=0.0", "x_0=10.0")
    )
    def test_monotony_z_negative(self, x_0):
        """
        Test monotony of the safe_log function with negative z values.

        Function :math:`g(x, z)` with :math:`x` fixed monotonically decreases
        on :math:`z < 0`. Without fixes to reduce the floating point errors,
        the safe_log function won't be monotone.
        """
        z = np.linspace(-1e-3, 0, 1001)
        z = z[z < 0]
        results = np.array([self.eval_safe_log(z_i, x_0) for z_i in z])
        assert (results[1:] < results[:-1]).all()

    @pytest.mark.parametrize(
        "x_0", (-10.0, 0.0, 10.0), ids=("x_0=-10.0", "x_0=0.0", "x_0=10.0")
    )
    def test_monotony_z_positive(self, x_0):
        """
        Test monotony of the safe_log function with positive z values.

        Function :math:`g(x, z)` with :math:`x` fixed monotonically increases
        on :math:`z > 0`. Without fixes to reduce the floating point errors,
        the safe_log function won't be monotone.
        """
        z = np.linspace(0, 1e-3, 1001)
        z = z[z > 0]
        results = np.array([self.eval_safe_log(z_i, x_0) for z_i in z])
        assert (results[1:] > results[:-1]).all()

    @pytest.mark.parametrize(
        "x_0", (-10.0, 0.0, 10.0), ids=("x_0=-10.0", "x_0=0.0", "x_0=10.0")
    )
    def test_symmetry(self, x_0):
        """
        Test symmetry of the function with respect to zero.

        Function :math:`g(x, z)` with :math:`x` fixed is symmetric with respect
        to :math:`z = 0`.
        """
        vmax = 1e-3
        z = np.linspace(-vmax, vmax, 1001)
        z = z[z != 0]
        results = np.array([self.eval_safe_log(z_i, x_0) for z_i in z])
        results_z_negative = results[z < 0]
        results_z_positive = results[z > 0]
        npt.assert_allclose(results_z_negative, results_z_positive[::-1])
