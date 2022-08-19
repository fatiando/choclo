# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import numpy.testing as npt
import pytest

from .._distance import distance_cartesian, distance_spherical, distance_spherical_core


class TestDistanceCartesian:
    @pytest.fixture(scope="class")
    def point_a(self):
        """
        Define a sample point a
        """
        return (1.1, 1.2, 1.3)

    @pytest.fixture(scope="class")
    def point_b(self):
        """
        Define a sample point b
        """
        return (1.1, 1.2, 2.4)

    def test_distance_cartesian(self, point_a, point_b):
        """
        Test if distance_cartesian works as expected
        """
        expected_value = 1.1
        npt.assert_allclose(distance_cartesian(point_a, point_b), expected_value)


class TestDistanceSpherical:
    @pytest.fixture(scope="class")
    def point_a(self):
        """
        Define a sample point a
        """
        return (32.3, 40.1, 1e4)

    @pytest.fixture(scope="class")
    def point_b(self):
        """
        Define a sample point b
        """
        return (32.3, 40.1, 1e4 + 100)

    def test_distance_spherical(self, point_a, point_b):
        """
        Test if distance_spherical works as expected
        """
        expected_value = 100
        npt.assert_allclose(distance_cartesian(point_a, point_b), expected_value)
