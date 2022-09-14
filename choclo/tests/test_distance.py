# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import numpy.testing as npt
import pytest

from ..utils import distance_cartesian, distance_spherical
from .utils import dumb_spherical_distance


@pytest.mark.parametrize(
    "point_a, point_b, expected_distance",
    [
        ((1.1, 1.2, 1.3), (2.4, 1.2, 1.3), 1.3),
        ((1.1, 1.2, 1.3), (1.1, -0.2, 1.3), 1.4),
        ((1.1, 1.2, 1.3), (1.1, 1.2, -2.4), 3.7),
        ((2.5, 3.4, -1.6), (8.7, -5.2, 0.4), 10.78888316740894),
    ],
)
def test_distance_cartesian(point_a, point_b, expected_distance):
    """
    Test if distance_cartesian works as expected
    """
    npt.assert_allclose(distance_cartesian(*point_a, *point_b), expected_distance)


class TestDistanceSpherical:
    """
    Tests for distance_spherical
    """

    @pytest.fixture(scope="class")
    def point_equator(self):
        """
        Define a point in the equator of a sphere of radius 2
        """
        return (45, 0, 2)

    @pytest.fixture(scope="class", params=[90, -90])
    def point_pole(self, request):
        """
        Define a point in the equator of a sphere of radius 2
        """
        latitude = request.param
        return (45, latitude, 2)

    @pytest.fixture(scope="class")
    def point_a(self):
        """
        Define a sample point in spherical coordinates
        """
        return 35.6, -40.8, 10.3

    @pytest.fixture(scope="class")
    def point_b(self):
        """
        Define a sample point in spherical coordinates
        """
        return 293.4, 70.9, 15.7

    def test_pole_and_equator(self, point_equator, point_pole):
        """
        Test spherical distance using points in the equator and in the pole
        """
        expected_distance = 2 ** (3 / 2)
        npt.assert_allclose(
            distance_spherical(*point_equator, *point_pole), expected_distance
        )

    def test_sample_points(self, point_a, point_b):
        """
        Test spherical distance using two sample points
        """
        npt.assert_allclose(
            distance_spherical(*point_a, *point_b),
            dumb_spherical_distance(point_a, point_b),
        )
