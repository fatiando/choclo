# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Tests for kernel functions of point sources
"""
import numpy as np
import numpy.testing as npt
import pytest

from .._point import (
    kernel_point_potential,
    kernel_point_g_easting,
    kernel_point_g_northing,
    kernel_point_g_upward,
    kernel_point_g_ee,
    kernel_point_g_nn,
    kernel_point_g_zz,
)
from .utils import NUMBA_IS_DISABLED


@pytest.fixture(name="sample_point_source")
def fixture_sample_point_source():
    """
    Return a sample point source
    """
    return (3.7, 4.3, -5.8)


@pytest.fixture(name="sample_coordinate")
def fixture_sample_coordinate():
    """
    Define a sample observation point
    """
    return 16.7, 13.2, 7.8


class TestSymmetryPotential:
    """
    Test the symmetry of the kernel for the potential of a point source
    """

    def build_points_in_sphere(self, radius, point):
        """
        Generate a set of observation points located in a sphere of radius
        ``radius`` and centered around the ``point``.
        """
        longitude = np.arange(0, 2 * np.pi, np.pi / 8)
        latitude = np.arange(-np.pi / 2, np.pi / 2, np.pi / 8)
        longitude, latitude = np.meshgrid(longitude, latitude)
        x = radius * np.cos(longitude) * np.cos(latitude)
        y = radius * np.sin(longitude) * np.cos(latitude)
        z = radius * np.sin(latitude)
        x += point[0]
        y += point[1]
        z += point[2]
        return tuple(coord.ravel() for coord in (x, y, z))

    def test_symmetry_on_sphere(self, sample_point_source):
        """
        Test the symmetry of the potential in points of a sphere
        """
        radius = 3.5
        observation_points = self.build_points_in_sphere(radius, sample_point_source)
        kernel_potential = [
            kernel_point_potential(*coords, *sample_point_source)
            for coords in zip(*observation_points)
        ]
        npt.assert_allclose(kernel_potential[0], kernel_potential)

    def test_potential_between_two_spheres(self, sample_point_source):
        """
        Test the potential between observation points in two spheres
        """
        radius_1, radius_2 = 3.5, 8.7
        sphere_1 = self.build_points_in_sphere(radius_1, sample_point_source)
        sphere_2 = self.build_points_in_sphere(radius_2, sample_point_source)
        kernel_potential_1 = np.array(
            [
                kernel_point_potential(*coords, *sample_point_source)
                for coords in zip(*sphere_1)
            ]
        )
        kernel_potential_2 = np.array(
            [
                kernel_point_potential(*coords, *sample_point_source)
                for coords in zip(*sphere_2)
            ]
        )
        npt.assert_allclose(
            kernel_potential_1, radius_2 / radius_1 * kernel_potential_2
        )

    @pytest.mark.skipif(not NUMBA_IS_DISABLED, reason="Numba is not disabled")
    def test_infinite_potential(self):
        """
        Test if we get an infinite kernel if the computation point is in the
        same location as the observation point.

        This test should be run only with Numba disabled.
        """
        point = (4.6, -8.9, -50.3)
        assert np.isinf(kernel_point_potential(*point, *point))

    @pytest.mark.skipif(NUMBA_IS_DISABLED, reason="Numba is disabled")
    def test_division_by_zero(self):
        """
        Test if we get a division by zero if the computation point is in the
        same location as the observation point.

        This test should be run only with Numba enabled.
        """
        point = (4.6, -8.9, -50.3)
        with pytest.raises(ZeroDivisionError):
            kernel_point_potential(*point, *point)


class TestSymmetryGradientEasting:
    """
    Test symmetry of the kernel for the g_easting component of a point source
    """

    @pytest.fixture
    def coords_northing_upward_plane(self, sample_point_source):
        """
        Define set of observation points in the northing-upward plane
        """
        # Define northing and upward coordinates (avoid the zero, that's where
        # the sample_point_source is located)
        northing = np.linspace(-11, 11, 12)
        upward = np.linspace(-7, 7, 8)
        # Generate meshgrid
        northing, upward = np.meshgrid(northing, upward)
        # Compute easting
        easting = np.zeros_like(northing)
        # Add the coordinates of the sample point source
        easting += sample_point_source[0]
        northing += sample_point_source[1]
        upward += sample_point_source[2]
        return easting, northing, upward

    @pytest.fixture
    def mirrored_points(self, sample_point_source):
        """
        Define two set of mirrored points across the northing-upward plane
        """
        # Define the northing and upward coordinates of the points
        northing = np.linspace(-11, 11, 27)
        upward = np.linspace(-38, 38, 23)
        # Define the easting coordinates for the first set
        easting_1 = np.linspace(1, 15, 15)
        # And mirror it for the second set
        easting_2 = -easting_1
        # Shift the coordinates to the sample_point_source
        easting_1 += sample_point_source[0]
        easting_2 += sample_point_source[0]
        northing += sample_point_source[1]
        upward += sample_point_source[2]
        return (easting_1, northing, upward), (easting_2, northing, upward)

    def test_zero_in_northing_upward(
        self, sample_point_source, coords_northing_upward_plane
    ):
        """
        Test if g_easting is zero in the northing-upward plane
        """
        g_easting = np.array(
            [
                kernel_point_g_easting(easting, northing, upward, *sample_point_source)
                for easting, northing, upward in zip(*coords_northing_upward_plane)
            ]
        )
        assert (g_easting <= 1e-30).all()

    def test_mirror_symmetry(self, sample_point_source, mirrored_points):
        """
        Test points with opposite easting coordinate
        """
        coords_1, coords_2 = mirrored_points
        g_easting_1 = np.array(
            [
                kernel_point_g_easting(easting, northing, upward, *sample_point_source)
                for easting, northing, upward in zip(*coords_1)
            ]
        )
        g_easting_2 = np.array(
            [
                kernel_point_g_easting(easting, northing, upward, *sample_point_source)
                for easting, northing, upward in zip(*coords_2)
            ]
        )
        npt.assert_allclose(g_easting_1, -g_easting_2)


class TestSymmetryGradientNorthing:
    """
    Test symmetry of the kernel for the g_northing component of a point source
    """

    @pytest.fixture
    def coords_easting_upward_plane(self, sample_point_source):
        """
        Define set of observation points in the easting-upward plane
        """
        # Define easting and upward coordinates (avoid the zero, that's where
        # the sample_point_source is located)
        easting = np.linspace(-11, 11, 12)
        upward = np.linspace(-7, 7, 8)
        # Generate meshgrid
        easting, upward = np.meshgrid(easting, upward)
        # Compute northing
        northing = np.zeros_like(easting)
        # Add the coordinates of the sample point source
        easting += sample_point_source[0]
        northing += sample_point_source[1]
        upward += sample_point_source[2]
        return easting, northing, upward

    @pytest.fixture
    def mirrored_points(self, sample_point_source):
        """
        Define two set of mirrored points across the easting-upward plane
        """
        # Define the easting and upward coordinates of the points
        easting = np.linspace(-11, 11, 27)
        upward = np.linspace(-38, 38, 23)
        # Define the northing coordinates for the first set
        northing_1 = np.linspace(1, 15, 15)
        # And mirror it for the second set
        northing_2 = -northing_1
        # Shift the coordinates to the sample_point_source
        easting += sample_point_source[0]
        northing_1 += sample_point_source[1]
        northing_2 += sample_point_source[1]
        upward += sample_point_source[2]
        return (easting, northing_1, upward), (easting, northing_2, upward)

    def test_zero_in_easting_upward(
        self, sample_point_source, coords_easting_upward_plane
    ):
        """
        Test if g_northing is zero in the easting-upward plane
        """
        g_northing = np.array(
            [
                kernel_point_g_northing(easting, northing, upward, *sample_point_source)
                for easting, northing, upward in zip(*coords_easting_upward_plane)
            ]
        )
        assert (g_northing <= 1e-30).all()

    def test_mirror_symmetry(self, sample_point_source, mirrored_points):
        """
        Test points with opposite northing coordinate
        """
        coords_1, coords_2 = mirrored_points
        g_northing_1 = np.array(
            [
                kernel_point_g_northing(easting, northing, upward, *sample_point_source)
                for easting, northing, upward in zip(*coords_1)
            ]
        )
        g_northing_2 = np.array(
            [
                kernel_point_g_northing(easting, northing, upward, *sample_point_source)
                for easting, northing, upward in zip(*coords_2)
            ]
        )
        npt.assert_allclose(g_northing_1, -g_northing_2)


class TestSymmetryGradientUpward:
    """
    Test symmetry of the kernel for the g_upward component of a point source
    """

    @pytest.fixture
    def coords_easting_northing_plane(self, sample_point_source):
        """
        Define set of observation points in the easting-northing plane
        """
        # Define easting and northing coordinates (avoid the zero, that's where
        # the sample_point_source is located)
        easting = np.linspace(-11, 11, 12)
        northing = np.linspace(-7, 7, 8)
        # Generate meshgrid
        easting, northing = np.meshgrid(easting, northing)
        # Compute upward
        upward = np.zeros_like(easting)
        # Add the coordinates of the sample point source
        easting += sample_point_source[0]
        northing += sample_point_source[1]
        upward += sample_point_source[2]
        return easting, northing, upward

    @pytest.fixture
    def mirrored_points(self, sample_point_source):
        """
        Define two set of mirrored points across the easting-northing plane
        """
        # Define the easting and upward coordinates of the points
        easting = np.linspace(-11, 11, 27)
        northing = np.linspace(-38, 38, 23)
        # Define the northing coordinates for the first set
        upward_1 = np.linspace(1, 15, 15)
        # And mirror it for the second set
        upward_2 = -upward_1
        # Shift the coordinates to the sample_point_source
        easting += sample_point_source[0]
        northing += sample_point_source[1]
        upward_1 += sample_point_source[2]
        upward_2 += sample_point_source[2]
        return (easting, northing, upward_1), (easting, northing, upward_2)

    def test_zero_in_easting_northing(
        self, sample_point_source, coords_easting_northing_plane
    ):
        """
        Test if g_upward is zero in the easting-upward plane
        """
        g_upward = np.array(
            [
                kernel_point_g_upward(easting, northing, upward, *sample_point_source)
                for easting, northing, upward in zip(*coords_easting_northing_plane)
            ]
        )
        assert (g_upward <= 1e-30).all()

    def test_mirror_symmetry(self, sample_point_source, mirrored_points):
        """
        Test points with opposite upward coordinate
        """
        coords_1, coords_2 = mirrored_points
        g_upward_1 = np.array(
            [
                kernel_point_g_upward(easting, northing, upward, *sample_point_source)
                for easting, northing, upward in zip(*coords_1)
            ]
        )
        g_upward_2 = np.array(
            [
                kernel_point_g_upward(easting, northing, upward, *sample_point_source)
                for easting, northing, upward in zip(*coords_2)
            ]
        )
        npt.assert_allclose(g_upward_1, -g_upward_2)


class TestGradientFiniteDifferences:
    """
    Test gradient kernels against finite-differences approximations of the
    potential
    """

    @pytest.fixture
    def finite_diff_g_easting(self, sample_coordinate, sample_point_source):
        """
        Compute g_easting through finite differences of the potential
        """
        easting_p, northing_p, upward_p = sample_coordinate
        easting_q, _, _ = sample_point_source
        # Compute a small increment in the easting coordinate
        d_easting = 1e-8 * (easting_p - easting_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p + d_easting, northing_p, upward_p)
        # Calculate g_easting through finite differences
        g_easting = (
            kernel_point_potential(*shifted_coordinate, *sample_point_source)
            - kernel_point_potential(*sample_coordinate, *sample_point_source)
        ) / d_easting
        return g_easting

    @pytest.fixture
    def finite_diff_g_northing(self, sample_coordinate, sample_point_source):
        """
        Compute g_northing through finite differences of the potential
        """
        easting_p, northing_p, upward_p = sample_coordinate
        _, northing_q, _ = sample_point_source
        # Compute a small increment in the easting coordinate
        d_northing = 1e-8 * (northing_p - northing_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p + d_northing, upward_p)
        # Calculate g_easting through finite differences
        g_northing = (
            kernel_point_potential(*shifted_coordinate, *sample_point_source)
            - kernel_point_potential(*sample_coordinate, *sample_point_source)
        ) / d_northing
        return g_northing

    @pytest.fixture
    def finite_diff_g_upward(self, sample_coordinate, sample_point_source):
        """
        Compute g_upward through finite differences of the potential
        """
        easting_p, northing_p, upward_p = sample_coordinate
        _, _, upward_q = sample_point_source
        # Compute a small increment in the easting coordinate
        d_upward = 1e-8 * (upward_p - upward_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p, upward_p + d_upward)
        # Calculate g_easting through finite differences
        g_upward = (
            kernel_point_potential(*shifted_coordinate, *sample_point_source)
            - kernel_point_potential(*sample_coordinate, *sample_point_source)
        ) / d_upward
        return g_upward

    def test_g_easting(
        self, sample_coordinate, sample_point_source, finite_diff_g_easting
    ):
        """
        Test kernel of g_easting against finite differences of the potential
        """
        npt.assert_allclose(
            finite_diff_g_easting,
            kernel_point_g_easting(*sample_coordinate, *sample_point_source),
        )

    def test_g_northing(
        self, sample_coordinate, sample_point_source, finite_diff_g_northing
    ):
        """
        Test kernel of g_northing against finite differences of the potential
        """
        npt.assert_allclose(
            finite_diff_g_northing,
            kernel_point_g_northing(*sample_coordinate, *sample_point_source),
        )

    def test_g_upward(
        self, sample_coordinate, sample_point_source, finite_diff_g_upward
    ):
        """
        Test kernel of g_upward against finite differences of the potential
        """
        npt.assert_allclose(
            finite_diff_g_upward,
            kernel_point_g_upward(*sample_coordinate, *sample_point_source),
        )


def test_laplacian(sample_coordinate, sample_point_source):
    """
    Test if diagonal tensor components satisfy Laplace's equation
    """
    g_ee = kernel_point_g_ee(*sample_coordinate, *sample_point_source)
    g_nn = kernel_point_g_nn(*sample_coordinate, *sample_point_source)
    g_zz = kernel_point_g_zz(*sample_coordinate, *sample_point_source)
    npt.assert_allclose(-g_zz, g_ee + g_nn)
