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

from ..point import (
    kernel_e,
    kernel_ee,
    kernel_en,
    kernel_eu,
    kernel_n,
    kernel_nn,
    kernel_nu,
    kernel_pot,
    kernel_u,
    kernel_uu,
)
from ..utils import distance_cartesian
from .utils import NUMBA_IS_DISABLED


def evaluate_kernel(coordinates, source, kernel):
    """
    Evaluate kernel on a set of observation points using a single source

    Parameters
    ----------
    coordinates : list
        Coordinates of the observation points in the following order:
            ``easting``, ``northing``, ``upward``
    source : tuple
        Tuple containing the coordinates of the point source
    kernel : func
        Kernel function that wants to be evaluated
    """
    result = []
    for coords in zip(*coordinates):
        distance = distance_cartesian(*coords, *source)
        result.append(kernel(*coords, *source, distance))
    return np.array(result)


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
    Test the symmetry of kernel_pot
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
        Test the symmetry of kernel_pot in points of a sphere
        """
        radius = 3.5
        observation_points = self.build_points_in_sphere(radius, sample_point_source)
        # Evaluate kernel_pot on every observation point
        kernel = evaluate_kernel(observation_points, sample_point_source, kernel_pot)
        # Check if all values are equal
        npt.assert_allclose(kernel[0], kernel)

    def test_potential_between_two_spheres(self, sample_point_source):
        """
        Test kernel_pot between observation points in two spheres
        """
        radius_1, radius_2 = 3.5, 8.7
        sphere_1 = self.build_points_in_sphere(radius_1, sample_point_source)
        sphere_2 = self.build_points_in_sphere(radius_2, sample_point_source)
        # Evaluate kernel_pot on every observation point
        kernel_1 = evaluate_kernel(sphere_1, sample_point_source, kernel_pot)
        kernel_2 = evaluate_kernel(sphere_2, sample_point_source, kernel_pot)
        # Check if all values are equal
        npt.assert_allclose(kernel_1, radius_2 / radius_1 * kernel_2)

    @pytest.mark.skipif(not NUMBA_IS_DISABLED, reason="Numba is not disabled")
    def test_infinite_potential(self):
        """
        Test if we get an infinite kernel_pot the computation point is in the
        same location as the observation point.

        This test should be run only with Numba disabled.
        """
        point = (4.6, -8.9, -50.3)
        distance = distance_cartesian(*point, *point)
        assert np.isinf(kernel_pot(*point, *point, distance))

    @pytest.mark.skipif(NUMBA_IS_DISABLED, reason="Numba is disabled")
    def test_division_by_zero(self):
        """
        Test if we get a division by zero if the computation point is in the
        same location as the observation point.

        This test should be run only with Numba enabled.
        """
        point = (4.6, -8.9, -50.3)
        distance = distance_cartesian(*point, *point)
        with pytest.raises(ZeroDivisionError):
            kernel_pot(*point, *point, distance)


class TestSymmetryKernelE:
    """
    Test symmetry of the kernel_e
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
        Test if kernel_e is zero in the northing-upward plane
        """
        kernel = evaluate_kernel(
            coords_northing_upward_plane, sample_point_source, kernel_e
        )
        assert (kernel <= 1e-30).all()

    def test_mirror_symmetry(self, sample_point_source, mirrored_points):
        """
        Test points with opposite easting coordinate
        """
        coords_1, coords_2 = mirrored_points
        kernel_1 = evaluate_kernel(coords_1, sample_point_source, kernel_e)
        kernel_2 = evaluate_kernel(coords_2, sample_point_source, kernel_e)
        npt.assert_allclose(kernel_1, -kernel_2)


class TestSymmetryKernelN:
    """
    Test symmetry of the kernel_n
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
        Test if kernel_n is zero in the easting-upward plane
        """
        kernel = evaluate_kernel(
            coords_easting_upward_plane, sample_point_source, kernel_n
        )
        assert (kernel <= 1e-30).all()

    def test_mirror_symmetry(self, sample_point_source, mirrored_points):
        """
        Test points with opposite northing coordinate
        """
        coords_1, coords_2 = mirrored_points
        kernel_1 = evaluate_kernel(coords_1, sample_point_source, kernel_n)
        kernel_2 = evaluate_kernel(coords_2, sample_point_source, kernel_n)
        npt.assert_allclose(kernel_1, -kernel_2)


class TestSymmetryKernelU:
    """
    Test symmetry of the kernel_u
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
        Test if kernel_u is zero in the easting-upward plane
        """
        kernel = evaluate_kernel(
            coords_easting_northing_plane, sample_point_source, kernel_u
        )
        assert (kernel <= 1e-30).all()

    def test_mirror_symmetry(self, sample_point_source, mirrored_points):
        """
        Test points with opposite upward coordinate
        """
        coords_1, coords_2 = mirrored_points
        kernel_1 = evaluate_kernel(coords_1, sample_point_source, kernel_u)
        kernel_2 = evaluate_kernel(coords_2, sample_point_source, kernel_u)
        npt.assert_allclose(kernel_1, -kernel_2)


class TestGradientFiniteDifferences:
    """
    Test gradient kernels against finite-differences approximations of the
    potential
    """

    # Define percentage for the finite difference displacement
    delta_percentage = 1e-8

    # Define expected relative error tolerance in the comparisons
    rtol = 1e-5

    @pytest.fixture
    def finite_diff_kernel_e(self, sample_coordinate, sample_point_source):
        """
        Compute kernel_e through finite differences of the kernel_pot
        """
        easting_p, northing_p, upward_p = sample_coordinate
        easting_q, _, _ = sample_point_source
        # Compute a small increment in the easting coordinate
        d_easting = self.delta_percentage * (easting_p - easting_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p + d_easting, northing_p, upward_p)
        # Calculate kernel_e through finite differences
        distance_shifted = distance_cartesian(*shifted_coordinate, *sample_point_source)
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        k_easting = (
            kernel_pot(*shifted_coordinate, *sample_point_source, distance_shifted)
            - kernel_pot(*sample_coordinate, *sample_point_source, distance)
        ) / d_easting
        return k_easting

    @pytest.fixture
    def finite_diff_kernel_n(self, sample_coordinate, sample_point_source):
        """
        Compute kernel_n through finite differences of the kernel_pot
        """
        easting_p, northing_p, upward_p = sample_coordinate
        _, northing_q, _ = sample_point_source
        # Compute a small increment in the easting coordinate
        d_northing = self.delta_percentage * (northing_p - northing_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p + d_northing, upward_p)
        # Calculate kernel_n through finite differences
        distance_shifted = distance_cartesian(*shifted_coordinate, *sample_point_source)
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        k_northing = (
            kernel_pot(*shifted_coordinate, *sample_point_source, distance_shifted)
            - kernel_pot(*sample_coordinate, *sample_point_source, distance)
        ) / d_northing
        return k_northing

    @pytest.fixture
    def finite_diff_kernel_u(self, sample_coordinate, sample_point_source):
        """
        Compute kernel_u through finite differences of the kernel_pot
        """
        easting_p, northing_p, upward_p = sample_coordinate
        _, _, upward_q = sample_point_source
        # Compute a small increment in the easting coordinate
        d_upward = self.delta_percentage * (upward_p - upward_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p, upward_p + d_upward)
        # Calculate kernel_u through finite differences
        distance_shifted = distance_cartesian(*shifted_coordinate, *sample_point_source)
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        k_upward = (
            kernel_pot(*shifted_coordinate, *sample_point_source, distance_shifted)
            - kernel_pot(*sample_coordinate, *sample_point_source, distance)
        ) / d_upward
        return k_upward

    def test_kernel_e(
        self, sample_coordinate, sample_point_source, finite_diff_kernel_e
    ):
        """
        Test kernel_e against finite differences of the kernel_pot
        """
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        npt.assert_allclose(
            finite_diff_kernel_e,
            kernel_e(*sample_coordinate, *sample_point_source, distance),
            rtol=self.rtol,
        )

    def test_kernel_n(
        self, sample_coordinate, sample_point_source, finite_diff_kernel_n
    ):
        """
        Test kernel_n against finite differences of the kernel_pot
        """
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        npt.assert_allclose(
            finite_diff_kernel_n,
            kernel_n(*sample_coordinate, *sample_point_source, distance),
            rtol=self.rtol,
        )

    def test_kernel_u(
        self, sample_coordinate, sample_point_source, finite_diff_kernel_u
    ):
        """
        Test kernel_u against finite differences of the kernel_pot
        """
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        npt.assert_allclose(
            finite_diff_kernel_u,
            kernel_u(*sample_coordinate, *sample_point_source, distance),
            rtol=self.rtol,
        )


class TestTensorFiniteDifferences:
    """
    Test tensor kernels against finite-differences approximations of the
    gradient
    """

    # Define percentage for the finite difference displacement
    delta_percentage = 1e-8

    # Define expected relative error tolerance in the comparisons
    rtol = 1e-5

    @pytest.fixture
    def finite_diff_kernel_ee(self, sample_coordinate, sample_point_source):
        """
        Compute kernel_ee through finite differences of the kernel_e
        """
        easting_p, northing_p, upward_p = sample_coordinate
        easting_q, _, _ = sample_point_source
        # Compute a small increment in the easting coordinate
        d_easting = self.delta_percentage * (easting_p - easting_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p + d_easting, northing_p, upward_p)
        # Calculate kernel_ee through finite differences
        distance_shifted = distance_cartesian(*shifted_coordinate, *sample_point_source)
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        kernel_ee = (
            kernel_e(*shifted_coordinate, *sample_point_source, distance_shifted)
            - kernel_e(*sample_coordinate, *sample_point_source, distance)
        ) / d_easting
        return kernel_ee

    @pytest.fixture
    def finite_diff_kernel_nn(self, sample_coordinate, sample_point_source):
        """
        Compute kernel_nn through finite differences of the kernel_n
        """
        easting_p, northing_p, upward_p = sample_coordinate
        _, northing_q, _ = sample_point_source
        # Compute a small increment in the easting coordinate
        d_northing = self.delta_percentage * (northing_p - northing_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p + d_northing, upward_p)
        # Calculate kernel_nn through finite differences
        distance_shifted = distance_cartesian(*shifted_coordinate, *sample_point_source)
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        kernel_nn = (
            kernel_n(*shifted_coordinate, *sample_point_source, distance_shifted)
            - kernel_n(*sample_coordinate, *sample_point_source, distance)
        ) / d_northing
        return kernel_nn

    @pytest.fixture
    def finite_diff_kernel_uu(self, sample_coordinate, sample_point_source):
        """
        Compute kernel_uu through finite differences of the kernel_u
        """
        easting_p, northing_p, upward_p = sample_coordinate
        _, _, upward_q = sample_point_source
        # Compute a small increment in the easting coordinate
        d_upward = self.delta_percentage * (upward_p - upward_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p, upward_p + d_upward)
        # Calculate kernel_uu through finite differences
        distance_shifted = distance_cartesian(*shifted_coordinate, *sample_point_source)
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        k_uu = (
            kernel_u(*shifted_coordinate, *sample_point_source, distance_shifted)
            - kernel_u(*sample_coordinate, *sample_point_source, distance)
        ) / d_upward
        return k_uu

    @pytest.fixture
    def finite_diff_kernel_en(self, sample_coordinate, sample_point_source):
        """
        Compute kernel_en through finite differences of the kernel_e
        """
        easting_p, northing_p, upward_p = sample_coordinate
        _, northing_q, _ = sample_point_source
        # Compute a small increment in the easting coordinate
        d_northing = self.delta_percentage * (northing_p - northing_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p + d_northing, upward_p)
        # Calculate kernel_en through finite differences
        distance_shifted = distance_cartesian(*shifted_coordinate, *sample_point_source)
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        k_en = (
            kernel_e(*shifted_coordinate, *sample_point_source, distance_shifted)
            - kernel_e(*sample_coordinate, *sample_point_source, distance)
        ) / d_northing
        return k_en

    @pytest.fixture
    def finite_diff_kernel_eu(self, sample_coordinate, sample_point_source):
        """
        Compute kernel_eu through finite differences of the kernel_e
        """
        easting_p, northing_p, upward_p = sample_coordinate
        _, _, upward_q = sample_point_source
        # Compute a small increment in the easting coordinate
        d_upward = self.delta_percentage * (upward_p - upward_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p, upward_p + d_upward)
        # Calculate kernel_eu through finite differences
        distance_shifted = distance_cartesian(*shifted_coordinate, *sample_point_source)
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        k_eu = (
            kernel_e(*shifted_coordinate, *sample_point_source, distance_shifted)
            - kernel_e(*sample_coordinate, *sample_point_source, distance)
        ) / d_upward
        return k_eu

    @pytest.fixture
    def finite_diff_kernel_nu(self, sample_coordinate, sample_point_source):
        """
        Test kernel_nu against finite differences of the kernel_n
        """
        easting_p, northing_p, upward_p = sample_coordinate
        _, _, upward_q = sample_point_source
        # Compute a small increment in the easting coordinate
        d_upward = self.delta_percentage * (upward_p - upward_q)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p, upward_p + d_upward)
        # Calculate k_nu through finite differences
        distance_shifted = distance_cartesian(*shifted_coordinate, *sample_point_source)
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        k_nu = (
            kernel_n(*shifted_coordinate, *sample_point_source, distance_shifted)
            - kernel_n(*sample_coordinate, *sample_point_source, distance)
        ) / d_upward
        return k_nu

    def test_g_ee(self, sample_coordinate, sample_point_source, finite_diff_kernel_ee):
        """
        Test kernel_ee against finite differences of the kernel_e
        """
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        npt.assert_allclose(
            finite_diff_kernel_ee,
            kernel_ee(*sample_coordinate, *sample_point_source, distance),
            rtol=self.rtol,
        )

    def test_g_nn(self, sample_coordinate, sample_point_source, finite_diff_kernel_nn):
        """
        Test kernel_nn against finite differences of the kernel_n
        """
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        npt.assert_allclose(
            finite_diff_kernel_nn,
            kernel_nn(*sample_coordinate, *sample_point_source, distance),
            rtol=self.rtol,
        )

    def test_g_zz(self, sample_coordinate, sample_point_source, finite_diff_kernel_uu):
        """
        Test kernel_uu against finite differences of the kernel_u
        """
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        npt.assert_allclose(
            finite_diff_kernel_uu,
            kernel_uu(*sample_coordinate, *sample_point_source, distance),
            rtol=self.rtol,
        )

    def test_g_en(self, sample_coordinate, sample_point_source, finite_diff_kernel_en):
        """
        Test kernel_en against finite differences of the kernel_e
        """
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        npt.assert_allclose(
            finite_diff_kernel_en,
            kernel_en(*sample_coordinate, *sample_point_source, distance),
            rtol=self.rtol,
        )

    def test_g_ez(self, sample_coordinate, sample_point_source, finite_diff_kernel_eu):
        """
        Test kernel_eu against finite differences of the kernel_e
        """
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        npt.assert_allclose(
            finite_diff_kernel_eu,
            kernel_eu(*sample_coordinate, *sample_point_source, distance),
            rtol=self.rtol,
        )

    def test_g_nz(self, sample_coordinate, sample_point_source, finite_diff_kernel_nu):
        """
        Test kernel_nu against finite differences of the kernel_n
        """
        distance = distance_cartesian(*sample_coordinate, *sample_point_source)
        npt.assert_allclose(
            finite_diff_kernel_nu,
            kernel_nu(*sample_coordinate, *sample_point_source, distance),
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

    @pytest.mark.parametrize("first_component", ("kernel_ee", "kernel_nn", "kernel_uu"))
    def test_laplacian(
        self,
        sample_observation_points,
        sample_point_source,
        first_component,
    ):
        """
        Test if diagonal tensor kernels satisfy Laplace's equation
        """
        k_ee = evaluate_kernel(
            sample_observation_points, sample_point_source, kernel_ee
        )
        k_nn = evaluate_kernel(
            sample_observation_points, sample_point_source, kernel_nn
        )
        k_uu = evaluate_kernel(
            sample_observation_points, sample_point_source, kernel_uu
        )
        # Set an atol to avoid getting failures when comparing values close to
        # zero
        atol = 1e-15
        rtol = 1e-7
        if first_component == "kernel_ee":
            npt.assert_allclose(-k_ee, k_nn + k_uu, rtol=rtol, atol=atol)
        if first_component == "kernel_nn":
            npt.assert_allclose(-k_nn, k_ee + k_uu, rtol=rtol, atol=atol)
        if first_component == "kernel_uu":
            npt.assert_allclose(-k_uu, k_ee + k_nn, rtol=rtol, atol=atol)
