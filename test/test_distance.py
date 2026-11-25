# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import boule as bl
import numpy as np
import numpy.testing as npt
import pytest
from utils import dumb_spherical_distance

from choclo.distance import distance_cartesian, distance_geodetic, distance_spherical


@pytest.mark.parametrize(
    ("point_a", "point_b", "expected_distance"),
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

    @pytest.fixture
    def point_equator(self):
        """
        Define a point in the equator of a sphere of radius 2
        """
        return (45, 0, 2)

    @pytest.fixture(params=[90, -90])
    def point_pole(self, request):
        """
        Define a point in the equator of a sphere of radius 2
        """
        latitude = request.param
        return (45, latitude, 2)

    @pytest.fixture
    def point_a(self):
        """
        Define a sample point in spherical coordinates
        """
        return 35.6, -40.8, 10.3

    @pytest.fixture
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


@pytest.mark.parametrize(
    "ellipsoid", [bl.WGS84, bl.GRS80, bl.EGM96, bl.Mars2009, bl.Ceres2018, bl.Vesta2017]
)
def test_distance_geodetic(ellipsoid):
    """
    Test if distance_geodetic works as expected
    """
    point_a = (-71.3, 33.5, 1e4)
    point_b = (-71.3, 33.5, 1e4 + 100)
    npt.assert_allclose(
        distance_geodetic(
            *point_a, *point_b, ellipsoid.semimajor_axis, ellipsoid.flattening
        ),
        100,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "ellipsoid", [bl.WGS84, bl.Mars2009, bl.Ceres2018, bl.Vesta2017]
)
def test_distance_geodetic_poles(ellipsoid):
    """
    Test if distance_geodetic works as expected when points are across the poles
    """
    point_a = (0, 90, 0)
    point_b = (0, -90, 0)
    npt.assert_allclose(
        distance_geodetic(
            *point_a, *point_b, ellipsoid.semimajor_axis, ellipsoid.flattening
        ),
        2 * ellipsoid.semiminor_axis,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "ellipsoid", [bl.WGS84, bl.Mars2009, bl.Ceres2018, bl.Vesta2017]
)
@pytest.mark.parametrize(
    "longitudes", [(0, 180), (90, 270), (-180, 0), (360, -180), (-90, 90)]
)
def test_distance_geodetic_equator(ellipsoid, longitudes):
    """
    Test if distance_geodetic works as expected when points are across the equator
    """
    point_a = (longitudes[0], 0, 0)
    point_b = (longitudes[1], 0, 0)
    npt.assert_allclose(
        distance_geodetic(
            *point_a, *point_b, ellipsoid.semimajor_axis, ellipsoid.flattening
        ),
        2 * ellipsoid.semimajor_axis,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "ellipsoid", [bl.WGS84, bl.Mars2009, bl.Ceres2018, bl.Vesta2017]
)
def test_distance_geodetic_equator_to_pole(ellipsoid):
    """
    Test if distance_geodetic works as expected when points are on the pole and equator
    """
    expected_distance = np.sqrt(
        ellipsoid.semimajor_axis**2 + ellipsoid.semiminor_axis**2
    )
    for latitude in (-90, 90):
        for longitude in np.linspace(0, 360, 36):
            npt.assert_allclose(
                distance_geodetic(
                    longitude,
                    0,
                    0,
                    0,
                    latitude,
                    0,
                    ellipsoid.semimajor_axis,
                    ellipsoid.flattening,
                ),
                expected_distance,
                rtol=1e-6,
            )
