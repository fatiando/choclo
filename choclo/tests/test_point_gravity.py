# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test gravity forward modelling functions for point sources
"""
import numpy as np
import numpy.testing as npt
import pytest

from ..point import (
    gravity_e,
    gravity_ee,
    gravity_en,
    gravity_eu,
    gravity_n,
    gravity_nn,
    gravity_nu,
    gravity_pot,
    gravity_u,
    gravity_uu,
)
from .utils import NUMBA_IS_DISABLED


@pytest.fixture(name="sample_point_source")
def fixture_sample_point_source():
    """
    Return a sample point source
    """
    return (3.7, 4.3, -5.8)


@pytest.fixture(name="sample_mass")
def fixture_sample_mass():
    """
    Return the mass for the sample point source
    """
    return 9e4


@pytest.fixture(name="sample_coordinate")
def fixture_sample_coordinate():
    """
    Define a sample observation point
    """
    return (16.7, 13.2, 7.8)


class TestSymmetryPotential:
    """
    Test the symmetry of gravity potential due to a point source
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

    def test_symmetry_on_sphere(self, sample_point_source, sample_mass):
        """
        Test the symmetry of the gravity potential in points of a sphere
        """
        radius = 3.5
        observation_points = self.build_points_in_sphere(radius, sample_point_source)
        # Evaluate gravity_pot on every observation point
        potential = tuple(
            gravity_pot(*coords, *sample_point_source, sample_mass)
            for coords in zip(*observation_points)
        )
        # Check if all values are equal
        npt.assert_allclose(potential[0], potential)

    def test_potential_between_two_spheres(self, sample_point_source, sample_mass):
        """
        Test the gravity potential between observation points in two spheres
        """
        radius_1, radius_2 = 3.5, 8.7
        sphere_1 = self.build_points_in_sphere(radius_1, sample_point_source)
        sphere_2 = self.build_points_in_sphere(radius_2, sample_point_source)
        # Evaluate kernel_pot on every observation point
        potential_1 = gravity_pot(*sphere_1, *sample_point_source, sample_mass)
        potential_2 = gravity_pot(*sphere_2, *sample_point_source, sample_mass)
        # Check if all values are equal
        npt.assert_allclose(potential_1, radius_2 / radius_1 * potential_2)

    @pytest.mark.skipif(not NUMBA_IS_DISABLED, reason="Numba is not disabled")
    def test_infinite_potential(self, sample_mass):
        """
        Test if we get an infinite kernel if the computation point is in the
        same location as the observation point.

        This test should be run only with Numba disabled.
        """
        point = (4.6, -8.9, -50.3)
        assert np.isinf(gravity_pot(*point, *point, sample_mass))

    @pytest.mark.skipif(NUMBA_IS_DISABLED, reason="Numba is disabled")
    def test_division_by_zero(self, sample_mass):
        """
        Test if we get a division by zero if the computation point is in the
        same location as the observation point.

        This test should be run only with Numba enabled.
        """
        point = (4.6, -8.9, -50.3)
        with pytest.raises(ZeroDivisionError):
            gravity_pot(*point, *point, sample_mass)


class TestSymmetryGravityE:
    """
    Test symmetry of the gravity_e
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
        self, sample_point_source, coords_northing_upward_plane, sample_mass
    ):
        """
        Test if kernel_e is zero in the northing-upward plane
        """
        g_e = np.array(
            list(
                gravity_e(*coords, *sample_point_source, sample_mass)
                for coords in zip(*coords_northing_upward_plane)
            )
        )
        assert (g_e <= 1e-30).all()

    def test_mirror_symmetry(self, sample_point_source, mirrored_points, sample_mass):
        """
        Test points with opposite easting coordinate
        """
        coords_1, coords_2 = mirrored_points
        g_e_1 = np.array(
            list(
                gravity_e(*coords, *sample_point_source, sample_mass)
                for coords in zip(*coords_1)
            )
        )
        g_e_2 = np.array(
            list(
                gravity_e(*coords, *sample_point_source, sample_mass)
                for coords in zip(*coords_2)
            )
        )
        npt.assert_allclose(g_e_1, -g_e_2)


class TestSymmetryGravityN:
    """
    Test symmetry of the gravity_n
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
        self, sample_point_source, coords_easting_upward_plane, sample_mass
    ):
        """
        Test if kernel_e is zero in the northing-upward plane
        """
        g_n = np.array(
            list(
                gravity_n(*coords, *sample_point_source, sample_mass)
                for coords in zip(*coords_easting_upward_plane)
            )
        )
        assert (g_n <= 1e-30).all()

    def test_mirror_symmetry(self, sample_point_source, mirrored_points, sample_mass):
        """
        Test points with opposite easting coordinate
        """
        coords_1, coords_2 = mirrored_points
        g_n_1 = np.array(
            list(
                gravity_n(*coords, *sample_point_source, sample_mass)
                for coords in zip(*coords_1)
            )
        )
        g_n_2 = np.array(
            list(
                gravity_n(*coords, *sample_point_source, sample_mass)
                for coords in zip(*coords_2)
            )
        )
        npt.assert_allclose(g_n_1, -g_n_2)


class TestSymmetryGravityU:
    """
    Test symmetry of the gravity_u
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
        self, sample_point_source, coords_easting_northing_plane, sample_mass
    ):
        """
        Test if gravity_u is zero in the easting-upward plane
        """
        g_u = np.array(
            list(
                gravity_u(*coords, *sample_point_source, sample_mass)
                for coords in zip(*coords_easting_northing_plane)
            )
        )
        assert (g_u <= 1e-30).all()

    def test_mirror_symmetry(self, sample_point_source, mirrored_points, sample_mass):
        """
        Test points with opposite upward coordinate
        """
        coords_1, coords_2 = mirrored_points
        g_u_1 = np.array(
            list(
                gravity_u(*coords, *sample_point_source, sample_mass)
                for coords in zip(*coords_1)
            )
        )
        g_u_2 = np.array(
            list(
                gravity_u(*coords, *sample_point_source, sample_mass)
                for coords in zip(*coords_2)
            )
        )
        npt.assert_allclose(g_u_1, -g_u_2)


class TestAccelerationFiniteDifferences:
    """
    Test acceleration components against finite-differences approximations of
    the gravity potential
    """

    # Define percentage for the finite difference displacement
    delta_percentage = 1e-8

    # Define expected relative error tolerance in the comparisons
    rtol = 1e-5

    @pytest.fixture
    def finite_diff_gravity_e(
        self, sample_coordinate, sample_point_source, sample_mass
    ):
        """
        Compute gravity_e through finite differences of the gravity_pot
        """
        easting_p, northing_p, upward_p = sample_coordinate
        easting_q, _, _ = sample_point_source
        # Compute a small increment in the easting coordinate
        d_easting = self.delta_percentage * (easting_p - easting_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p + d_easting, northing_p, upward_p)
        # Calculate g_e through finite differences
        g_e = (
            gravity_pot(*shifted_coordinate, *sample_point_source, sample_mass)
            - gravity_pot(*sample_coordinate, *sample_point_source, sample_mass)
        ) / d_easting
        return g_e

    @pytest.fixture
    def finite_diff_gravity_n(
        self, sample_coordinate, sample_point_source, sample_mass
    ):
        """
        Compute gravity_n through finite differences of the gravity_pot
        """
        easting_p, northing_p, upward_p = sample_coordinate
        _, northing_q, _ = sample_point_source
        # Compute a small increment in the easting coordinate
        d_northing = self.delta_percentage * (northing_p - northing_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p + d_northing, upward_p)
        # Calculate g_n through finite differences
        g_n = (
            gravity_pot(*shifted_coordinate, *sample_point_source, sample_mass)
            - gravity_pot(*sample_coordinate, *sample_point_source, sample_mass)
        ) / d_northing
        return g_n

    @pytest.fixture
    def finite_diff_gravity_u(
        self, sample_coordinate, sample_point_source, sample_mass
    ):
        """
        Compute gravity_u through finite differences of the gravity_pot
        """
        easting_p, northing_p, upward_p = sample_coordinate
        _, _, upward_q = sample_point_source
        # Compute a small increment in the easting coordinate
        d_upward = self.delta_percentage * (upward_p - upward_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p, upward_p + d_upward)
        # Calculate g_u through finite differences
        g_u = (
            gravity_pot(*shifted_coordinate, *sample_point_source, sample_mass)
            - gravity_pot(*sample_coordinate, *sample_point_source, sample_mass)
        ) / d_upward
        return g_u

    def test_gravity_e(
        self, sample_coordinate, sample_point_source, finite_diff_gravity_e, sample_mass
    ):
        """
        Test gravity_e against finite differences of the gravity_pot
        """
        npt.assert_allclose(
            finite_diff_gravity_e,
            gravity_e(*sample_coordinate, *sample_point_source, sample_mass),
            rtol=self.rtol,
        )

    def test_gravity_n(
        self, sample_coordinate, sample_point_source, finite_diff_gravity_n, sample_mass
    ):
        """
        Test gravity_n against finite differences of the gravity_pot
        """
        npt.assert_allclose(
            finite_diff_gravity_n,
            gravity_n(*sample_coordinate, *sample_point_source, sample_mass),
            rtol=self.rtol,
        )

    def test_gravity_u(
        self, sample_coordinate, sample_point_source, finite_diff_gravity_u, sample_mass
    ):
        """
        Test gravity_u against finite differences of the gravity_pot
        """
        npt.assert_allclose(
            finite_diff_gravity_u,
            gravity_u(*sample_coordinate, *sample_point_source, sample_mass),
            rtol=self.rtol,
        )


class TestTensorFiniteDifferences:
    """
    Test tensor gravity functions against finite-differences approximations of
    the gradient components
    """

    # Define percentage for the finite difference displacement
    delta_percentage = 1e-8

    # Define expected relative error tolerance in the comparisons
    rtol = 1e-5

    @pytest.fixture
    def finite_diff_gravity_ee(
        self, sample_coordinate, sample_point_source, sample_mass
    ):
        """
        Compute gravity_ee through finite differences of the gravity_e
        """
        easting_p, northing_p, upward_p = sample_coordinate
        easting_q, _, _ = sample_point_source
        # Compute a small increment in the easting coordinate
        d_easting = self.delta_percentage * (easting_p - easting_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p + d_easting, northing_p, upward_p)
        # Calculate g_ee through finite differences
        g_ee = (
            gravity_e(*shifted_coordinate, *sample_point_source, sample_mass)
            - gravity_e(*sample_coordinate, *sample_point_source, sample_mass)
        ) / d_easting
        return g_ee

    @pytest.fixture
    def finite_diff_gravity_nn(
        self, sample_coordinate, sample_point_source, sample_mass
    ):
        """
        Compute gravity_nn through finite differences of the gravity_n
        """
        easting_p, northing_p, upward_p = sample_coordinate
        _, northing_q, _ = sample_point_source
        # Compute a small increment in the easting coordinate
        d_northing = self.delta_percentage * (northing_p - northing_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p + d_northing, upward_p)
        # Calculate g_nn through finite differences
        g_nn = (
            gravity_n(*shifted_coordinate, *sample_point_source, sample_mass)
            - gravity_n(*sample_coordinate, *sample_point_source, sample_mass)
        ) / d_northing
        return g_nn

    @pytest.fixture
    def finite_diff_gravity_uu(
        self, sample_coordinate, sample_point_source, sample_mass
    ):
        """
        Compute gravity_uu through finite differences of the gravity_u
        """
        easting_p, northing_p, upward_p = sample_coordinate
        _, _, upward_q = sample_point_source
        # Compute a small increment in the easting coordinate
        d_upward = self.delta_percentage * (upward_p - upward_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p, upward_p + d_upward)
        # Calculate g_uu through finite differences
        g_uu = (
            gravity_u(*shifted_coordinate, *sample_point_source, sample_mass)
            - gravity_u(*sample_coordinate, *sample_point_source, sample_mass)
        ) / d_upward
        return g_uu

    @pytest.fixture
    def finite_diff_gravity_en(
        self, sample_coordinate, sample_point_source, sample_mass
    ):
        """
        Compute gravity_en through finite differences of the gravity_e
        """
        easting_p, northing_p, upward_p = sample_coordinate
        _, northing_q, _ = sample_point_source
        # Compute a small increment in the easting coordinate
        d_northing = self.delta_percentage * (northing_p - northing_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p + d_northing, upward_p)
        # Calculate g_en through finite differences
        g_en = (
            gravity_e(*shifted_coordinate, *sample_point_source, sample_mass)
            - gravity_e(*sample_coordinate, *sample_point_source, sample_mass)
        ) / d_northing
        return g_en

    @pytest.fixture
    def finite_diff_gravity_eu(
        self, sample_coordinate, sample_point_source, sample_mass
    ):
        """
        Compute gravity_eu through finite differences of the gravity_e
        """
        easting_p, northing_p, upward_p = sample_coordinate
        _, _, upward_q = sample_point_source
        # Compute a small increment in the easting coordinate
        d_upward = self.delta_percentage * (upward_p - upward_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p, upward_p + d_upward)
        # Calculate g_eu through finite differences
        g_eu = (
            gravity_e(*shifted_coordinate, *sample_point_source, sample_mass)
            - gravity_e(*sample_coordinate, *sample_point_source, sample_mass)
        ) / d_upward
        return g_eu

    @pytest.fixture
    def finite_diff_gravity_nu(
        self, sample_coordinate, sample_point_source, sample_mass
    ):
        """
        Test gravity_nu against finite differences of the gravity_n
        """
        easting_p, northing_p, upward_p = sample_coordinate
        _, _, upward_q = sample_point_source
        # Compute a small increment in the easting coordinate
        d_upward = self.delta_percentage * (upward_p - upward_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p, upward_p + d_upward)
        # Calculate g_nu through finite differences
        g_nu = (
            gravity_n(*shifted_coordinate, *sample_point_source, sample_mass)
            - gravity_n(*sample_coordinate, *sample_point_source, sample_mass)
        ) / d_upward
        return g_nu

    def test_gravity_ee(
        self,
        sample_coordinate,
        sample_point_source,
        finite_diff_gravity_ee,
        sample_mass,
    ):
        """
        Test gravity_ee against finite differences of the gravity_e
        """
        npt.assert_allclose(
            finite_diff_gravity_ee,
            gravity_ee(*sample_coordinate, *sample_point_source, sample_mass),
            rtol=self.rtol,
        )

    def test_gravity_nn(
        self,
        sample_coordinate,
        sample_point_source,
        finite_diff_gravity_nn,
        sample_mass,
    ):
        """
        Test gravity_nn against finite differences of the gravity_n
        """
        npt.assert_allclose(
            finite_diff_gravity_nn,
            gravity_nn(*sample_coordinate, *sample_point_source, sample_mass),
            rtol=self.rtol,
        )

    def test_gravity_uu(
        self,
        sample_coordinate,
        sample_point_source,
        finite_diff_gravity_uu,
        sample_mass,
    ):
        """
        Test gravity_uu against finite differences of the gravity_u
        """
        npt.assert_allclose(
            finite_diff_gravity_uu,
            gravity_uu(*sample_coordinate, *sample_point_source, sample_mass),
            rtol=self.rtol,
        )

    def test_gravity_en(
        self,
        sample_coordinate,
        sample_point_source,
        finite_diff_gravity_en,
        sample_mass,
    ):
        """
        Test gravity_en against finite differences of the gravity_e
        """
        npt.assert_allclose(
            finite_diff_gravity_en,
            gravity_en(*sample_coordinate, *sample_point_source, sample_mass),
            rtol=self.rtol,
        )

    def test_gravity_eu(
        self,
        sample_coordinate,
        sample_point_source,
        finite_diff_gravity_eu,
        sample_mass,
    ):
        """
        Test gravity_eu against finite differences of the gravity_e
        """
        npt.assert_allclose(
            finite_diff_gravity_eu,
            gravity_eu(*sample_coordinate, *sample_point_source, sample_mass),
            rtol=self.rtol,
        )

    def test_gravity_nu(
        self,
        sample_coordinate,
        sample_point_source,
        finite_diff_gravity_nu,
        sample_mass,
    ):
        """
        Test gravity_nu against finite differences of the gravity_n
        """
        npt.assert_allclose(
            finite_diff_gravity_nu,
            gravity_nu(*sample_coordinate, *sample_point_source, sample_mass),
            rtol=self.rtol,
        )


class TestLaplacian:
    @pytest.fixture
    def sample_observation_points(self, sample_point_source):
        """
        Define a 3D grid of observation points around the sample point source.
        The grid doesn't contain an observation point located in the same
        location as the point source.
        """
        # Build the observation points
        easting = np.linspace(-10, 10, 21)
        northing = np.linspace(-10, 10, 21)
        upward = np.linspace(-10, 10, 21)
        easting, northing, upward = tuple(
            a.ravel() for a in np.meshgrid(easting, northing, upward)
        )
        # Remove the location of the sample point source
        point_source_location = (easting == 0) & (northing == 0) & (upward == 0)
        easting, northing, upward = tuple(
            a[np.logical_not(point_source_location)]
            for a in (easting, northing, upward)
        )
        # Shift the coordinates
        easting += sample_point_source[0]
        northing += sample_point_source[1]
        upward += sample_point_source[2]
        return (easting, northing, upward)

    @pytest.mark.parametrize("first_component", ("g_ee", "g_nn", "g_uu"))
    def test_laplacian(
        self,
        sample_observation_points,
        sample_point_source,
        sample_mass,
        first_component,
    ):
        """
        Test if diagonal tensor functions satisfy Laplace's equation
        """
        g_ee = np.array(
            [
                gravity_ee(e, n, u, *sample_point_source, sample_mass)
                for e, n, u in zip(*sample_observation_points)
            ]
        )
        g_nn = np.array(
            [
                gravity_nn(e, n, u, *sample_point_source, sample_mass)
                for e, n, u in zip(*sample_observation_points)
            ]
        )
        g_uu = np.array(
            [
                gravity_uu(e, n, u, *sample_point_source, sample_mass)
                for e, n, u in zip(*sample_observation_points)
            ]
        )
        # Set an atol to avoid getting failures when comparing values close to
        # zero
        atol = 1e-21
        rtol = 1e-7
        if first_component == "g_ee":
            npt.assert_allclose(-g_ee, g_nn + g_uu, rtol=rtol, atol=atol)
        if first_component == "g_nn":
            npt.assert_allclose(-g_nn, g_ee + g_uu, rtol=rtol, atol=atol)
        if first_component == "g_uu":
            npt.assert_allclose(-g_uu, g_ee + g_nn, rtol=rtol, atol=atol)
