# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test magnetic forward modelling functions for rectangular prisms
"""
import numpy as np
import numpy.testing as npt
import pytest

from ..prism import magnetic_e, magnetic_field, magnetic_n, magnetic_u


@pytest.fixture(name="sample_prism")
def fixture_sample_prism():
    """
    Return a sample prism
    """
    return np.array([110.2, 130.5, -40.5, -31.3, -1014.5, -994.3])


@pytest.fixture(name="sample_magnetization")
def fixture_sample_magnetization():
    """
    Return a sample magnetization vector
    """
    return np.array([314.3, -512.5, 256.9])


@pytest.fixture(name="sample_3d_grid")
def fixture_sample_3d_grid(sample_prism):
    """
    Return a set of observation points in a 3d grid

    These observation points are built around the center of the sample prism,
    but they don't include points on its center, its edges, faces or vertices.
    """
    easting = np.linspace(-40, 40, 11)
    northing = np.linspace(-40, 40, 11)
    upward = np.linspace(-40, 40, 11)
    # Compute meshgrid
    easting, northing, upward = np.meshgrid(easting, northing, upward)
    # Remove the point that falls in the center of the sample prism
    is_prism_center = (easting == 0) & (northing == 0) & (upward == 0)
    easting = easting[np.logical_not(is_prism_center)]
    northing = northing[np.logical_not(is_prism_center)]
    upward = upward[np.logical_not(is_prism_center)]
    # Shift coordinates
    prism_center_easting, prism_center_northing, prism_center_upward = get_prism_center(
        sample_prism
    )
    easting += prism_center_easting
    northing += prism_center_northing
    upward += prism_center_upward
    return easting, northing, upward


def get_prism_center(prism):
    """
    Return the center of the prism
    """
    easting = (prism[0] + prism[1]) / 2
    northing = (prism[2] + prism[3]) / 2
    upward = (prism[4] + prism[5]) / 2
    return easting, northing, upward


class TestSymmetryBe:
    """
    Test symmetry of easting component of the magnetic field
    """

    atol = 1e-17  # absolute tolerance for values near zero

    @pytest.mark.parametrize(
        "magnetization",
        [
            (500, 0, 0),
            (-500, 0, 0),
            (0, 500, 0),
            (0, -500, 0),
            (0, 0, 500),
            (0, 0, -500),
        ],
    )
    def test_symmetry_across_easting_northing(
        self, sample_3d_grid, sample_prism, magnetization
    ):
        """
        Test symmetry of magnetic_e across the easting-northing plane that
        passes through the center of the prism
        """
        easting, northing, upward = sample_3d_grid
        # Keep only the observation points that are above the prism center
        _, _, prism_center_u = get_prism_center(sample_prism)
        is_top = upward > prism_center_u
        easting = easting[is_top]
        northing = northing[is_top]
        upward_top = upward[is_top]
        # Create a symmetrical upward coordinate for points below the prism
        # center
        upward_bottom = 2 * prism_center_u - upward_top
        # Compute magnetic_e on every observation point
        magnetization = np.array(magnetization)
        b_e_top = np.array(
            list(
                magnetic_e(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_top.ravel()
                )
            )
        )
        b_e_bottom = np.array(
            list(
                magnetic_e(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_bottom.ravel()
                )
            )
        )
        # Check symmetry between top and bottom
        if magnetization[2] != 0:
            npt.assert_allclose(b_e_top, -b_e_bottom, atol=self.atol)
        else:
            npt.assert_allclose(b_e_top, b_e_bottom, atol=self.atol)

    @pytest.mark.parametrize(
        "magnetization",
        [
            (500, 0, 0),
            (-500, 0, 0),
            (0, 500, 0),
            (0, -500, 0),
            (0, 0, 500),
            (0, 0, -500),
        ],
    )
    def test_symmetry_across_easting_upward(
        self, sample_3d_grid, sample_prism, magnetization
    ):
        """
        Test symmetry of magnetic_e across the easting-upward plane that
        passes through the center of the prism
        """
        easting, northing, upward = sample_3d_grid
        # Keep only the observation points that are north the prism center
        _, prism_center_n, _ = get_prism_center(sample_prism)
        is_north = northing > prism_center_n
        easting = easting[is_north]
        northing_north = northing[is_north]
        upward = upward[is_north]
        # Create a symmetrical upward coordinate for points south the prism
        # center
        northing_south = 2 * prism_center_n - northing_north
        # Compute magnetic_e on every observation point
        magnetization = np.array(magnetization)
        b_e_north = np.array(
            list(
                magnetic_e(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting.ravel(), northing_north.ravel(), upward.ravel()
                )
            )
        )
        b_e_south = np.array(
            list(
                magnetic_e(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting.ravel(), northing_south.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between south and north
        if magnetization[1] != 0:
            npt.assert_allclose(b_e_north, -b_e_south, atol=self.atol)
        else:
            npt.assert_allclose(b_e_north, b_e_south, atol=self.atol)

    @pytest.mark.parametrize(
        "magnetization",
        [
            (500, 0, 0),
            (-500, 0, 0),
            (0, 500, 0),
            (0, -500, 0),
            (0, 0, 500),
            (0, 0, -500),
        ],
    )
    def test_symmetry_across_northing_upward(
        self, sample_3d_grid, sample_prism, magnetization
    ):
        """
        Test symmetry of magnetic_e across the northing-upward plane that
        passes through the center of the prism
        """
        easting, northing, upward = sample_3d_grid
        # Keep only the observation points that are east the prism center
        prism_center_e, _, _ = get_prism_center(sample_prism)
        is_east = easting > prism_center_e
        easting_east = easting[is_east]
        northing = northing[is_east]
        upward = upward[is_east]
        # Create a symmetrical upward coordinate for points west the prism
        # center
        easting_west = 2 * prism_center_e - easting_east
        # Compute magnetic_e on every observation point
        magnetization = np.array(magnetization)
        b_e_east = np.array(
            list(
                magnetic_e(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting_east.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        b_e_west = np.array(
            list(
                magnetic_e(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting_west.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between west and east
        if magnetization[0] != 0:
            npt.assert_allclose(b_e_east, b_e_west, atol=self.atol)
        else:
            npt.assert_allclose(b_e_east, -b_e_west, atol=self.atol)

    def test_symmetry_when_flipping(self, sample_3d_grid, sample_prism):
        """
        Test symmetry of magnetic_e when flipping its direction
        """
        easting, northing, upward = sample_3d_grid
        # Keep only points with easting on east of the prism center
        prism_center_e, _, _ = get_prism_center(sample_prism)
        is_east_or_equal = easting >= prism_center_e
        easting = easting[is_east_or_equal]
        northing = northing[is_east_or_equal]
        upward = upward[is_east_or_equal]
        # Define two magnetic moments
        magnetization_east = np.array([500.0, 0, 0])
        magnetization_west = np.array([-500.0, 0, 0])
        # Compute the magnetic field generated by each moment on the
        # observation points
        b_e_east = np.array(
            list(
                magnetic_e(e, n, u, sample_prism, magnetization_east)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        b_e_west = np.array(
            list(
                magnetic_e(e, n, u, sample_prism, magnetization_west)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        # Check if the sign gets inverted
        npt.assert_allclose(b_e_east, -b_e_west)


class TestSymmetryBn:
    """
    Test symmetry of easting component of the magnetic field
    """

    atol = 1e-17  # absolute tolerance for values near zero

    @pytest.mark.parametrize(
        "magnetization",
        [
            (500, 0, 0),
            (-500, 0, 0),
            (0, 500, 0),
            (0, -500, 0),
            (0, 0, 500),
            (0, 0, -500),
        ],
    )
    def test_symmetry_across_easting_northing(
        self, sample_3d_grid, sample_prism, magnetization
    ):
        """
        Test symmetry of magnetic_n across the easting-northing plane that
        passes through the center of the prism
        """
        easting, northing, upward = sample_3d_grid
        # Keep only the observation points that are above the prism center
        _, _, prism_center_u = get_prism_center(sample_prism)
        is_top = upward > prism_center_u
        easting = easting[is_top]
        northing = northing[is_top]
        upward_top = upward[is_top]
        # Create a symmetrical upward coordinate for points below the prism
        # center
        upward_bottom = 2 * prism_center_u - upward_top
        # Compute magnetic_n on every observation point
        magnetization = np.array(magnetization)
        b_n_top = np.array(
            list(
                magnetic_n(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_top.ravel()
                )
            )
        )
        b_n_bottom = np.array(
            list(
                magnetic_n(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_bottom.ravel()
                )
            )
        )
        # Check symmetry between top and bottom
        if magnetization[2] != 0:
            npt.assert_allclose(b_n_top, -b_n_bottom, atol=self.atol)
        else:
            npt.assert_allclose(b_n_top, b_n_bottom, atol=self.atol)

    @pytest.mark.parametrize(
        "magnetization",
        [
            (500, 0, 0),
            (-500, 0, 0),
            (0, 500, 0),
            (0, -500, 0),
            (0, 0, 500),
            (0, 0, -500),
        ],
    )
    def test_symmetry_across_easting_upward(
        self, sample_3d_grid, sample_prism, magnetization
    ):
        """
        Test symmetry of magnetic_n across the easting-upward plane that
        passes through the center of the prism
        """
        easting, northing, upward = sample_3d_grid
        # Keep only the observation points that are north the prism center
        _, prism_center_n, _ = get_prism_center(sample_prism)
        is_north = northing > prism_center_n
        easting = easting[is_north]
        northing_north = northing[is_north]
        upward = upward[is_north]
        # Create a symmetrical upward coordinate for points south the prism
        # center
        northing_south = 2 * prism_center_n - northing_north
        # Compute magnetic_n on every observation point
        magnetization = np.array(magnetization)
        b_n_north = np.array(
            list(
                magnetic_n(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting.ravel(), northing_north.ravel(), upward.ravel()
                )
            )
        )
        b_n_south = np.array(
            list(
                magnetic_n(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting.ravel(), northing_south.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between south and north
        if magnetization[1] != 0:
            npt.assert_allclose(b_n_north, b_n_south, atol=self.atol)
        else:
            npt.assert_allclose(b_n_north, -b_n_south, atol=self.atol)

    @pytest.mark.parametrize(
        "magnetization",
        [
            (500, 0, 0),
            (-500, 0, 0),
            (0, 500, 0),
            (0, -500, 0),
            (0, 0, 500),
            (0, 0, -500),
        ],
    )
    def test_symmetry_across_northing_upward(
        self, sample_3d_grid, sample_prism, magnetization
    ):
        """
        Test symmetry of magnetic_n across the northing-upward plane that
        passes through the center of the prism
        """
        easting, northing, upward = sample_3d_grid
        # Keep only the observation points that are east the prism center
        prism_center_e, _, _ = get_prism_center(sample_prism)
        is_east = easting > prism_center_e
        easting_east = easting[is_east]
        northing = northing[is_east]
        upward = upward[is_east]
        # Create a symmetrical upward coordinate for points west the prism
        # center
        easting_west = 2 * prism_center_e - easting_east
        # Compute magnetic_n on every observation point
        magnetization = np.array(magnetization)
        b_n_east = np.array(
            list(
                magnetic_n(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting_east.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        b_n_west = np.array(
            list(
                magnetic_n(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting_west.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between west and east
        if magnetization[0] != 0:
            npt.assert_allclose(b_n_east, -b_n_west, atol=self.atol)
        else:
            npt.assert_allclose(b_n_east, b_n_west, atol=self.atol)

    def test_symmetry_when_flipping(self, sample_3d_grid, sample_prism):
        """
        Test symmetry of magnetic_n when flipping its direction
        """
        easting, northing, upward = sample_3d_grid
        # Keep only points with northing on north of the dipole
        _, prism_center_n, _ = get_prism_center(sample_prism)
        is_north_or_equal = northing >= prism_center_n
        easting = easting[is_north_or_equal]
        northing = northing[is_north_or_equal]
        upward = upward[is_north_or_equal]
        # Define two magnetic moments
        magnetization_north = np.array([0, 500.0, 0])
        magnetization_south = np.array([0, -500.0, 0])
        # Compute the magnetic field generated by each moment on the
        # observation points
        b_n_north = np.array(
            list(
                magnetic_n(e, n, u, sample_prism, magnetization_north)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        b_n_south = np.array(
            list(
                magnetic_n(e, n, u, sample_prism, magnetization_south)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        # Check if the sign gets inverted
        npt.assert_allclose(b_n_north, -b_n_south)


class TestSymmetryBu:
    """
    Test symmetry of upward component of the magnetic field
    """

    atol = 1e-17  # absolute tolerance for values near zero

    @pytest.mark.parametrize(
        "magnetization",
        [
            (500, 0, 0),
            (-500, 0, 0),
            (0, 500, 0),
            (0, -500, 0),
            (0, 0, 500),
            (0, 0, -500),
        ],
    )
    def test_symmetry_across_easting_northing(
        self, sample_3d_grid, sample_prism, magnetization
    ):
        """
        Test symmetry of magnetic_e across the easting-northing plane that
        passes through the center of the prism
        """
        easting, northing, upward = sample_3d_grid
        # Keep only observation points that are above the center of the prism
        _, _, prism_center_u = get_prism_center(sample_prism)
        is_top = upward > prism_center_u
        easting = easting[is_top]
        northing = northing[is_top]
        upward_top = upward[is_top]
        # Create a symmetrical upward coordinate for points below the prism
        # center
        upward_bottom = 2 * prism_center_u - upward_top
        # Compute magnetic_u on every observation point
        magnetization = np.array(magnetization)
        b_u_top = np.array(
            list(
                magnetic_u(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_top.ravel()
                )
            )
        )
        b_u_bottom = np.array(
            list(
                magnetic_u(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_bottom.ravel()
                )
            )
        )
        # Check symmetry between top and bottom
        if magnetization[2] != 0:
            npt.assert_allclose(b_u_top, b_u_bottom, atol=self.atol)
        else:
            npt.assert_allclose(b_u_top, -b_u_bottom, atol=self.atol)

    @pytest.mark.parametrize(
        "magnetization",
        [
            (500, 0, 0),
            (-500, 0, 0),
            (0, 500, 0),
            (0, -500, 0),
            (0, 0, 500),
            (0, 0, -500),
        ],
    )
    def test_symmetry_across_easting_upward(
        self, sample_3d_grid, sample_prism, magnetization
    ):
        """
        Test symmetry of magnetic_e across the easting-upward plane that
        passes through the center of the prism
        """
        easting, northing, upward = sample_3d_grid
        # Keep only the observation points that are north of the prism center
        _, prism_center_n, _ = get_prism_center(sample_prism)
        is_north = northing > prism_center_n
        easting = easting[is_north]
        northing_north = northing[is_north]
        upward = upward[is_north]
        # Create a symmetrical upward coordinate for points south the prism
        # center
        northing_south = 2 * prism_center_n - northing_north
        # Compute magnetic_u on every observation point
        magnetization = np.array(magnetization)
        b_u_north = np.array(
            list(
                magnetic_u(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting.ravel(), northing_north.ravel(), upward.ravel()
                )
            )
        )
        b_u_south = np.array(
            list(
                magnetic_u(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting.ravel(), northing_south.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between south and north
        if magnetization[1] != 0:
            npt.assert_allclose(b_u_north, -b_u_south, atol=self.atol)
        else:
            npt.assert_allclose(b_u_north, b_u_south, atol=self.atol)

    @pytest.mark.parametrize(
        "magnetization",
        [
            (500, 0, 0),
            (-500, 0, 0),
            (0, 500, 0),
            (0, -500, 0),
            (0, 0, 500),
            (0, 0, -500),
        ],
    )
    def test_symmetry_across_northing_upward(
        self, sample_3d_grid, sample_prism, magnetization
    ):
        """
        Test symmetry of magnetic_e across the northing-upward plane that
        passes through the center of the prism
        """
        easting, northing, upward = sample_3d_grid
        # Keep only the observation points that are east of the prism center
        prism_center_e, _, _ = get_prism_center(sample_prism)
        is_east = easting > prism_center_e
        easting_east = easting[is_east]
        northing = northing[is_east]
        upward = upward[is_east]
        # Create a symmetrical upward coordinate for points west the center of
        # the prism
        easting_west = 2 * prism_center_e - easting_east
        # Compute magnetic_u on every observation point
        magnetization = np.array(magnetization)
        b_u_east = np.array(
            list(
                magnetic_u(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting_east.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        b_u_west = np.array(
            list(
                magnetic_u(e, n, u, sample_prism, magnetization)
                for e, n, u in zip(
                    easting_west.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between east and west
        if magnetization[0] != 0:
            npt.assert_allclose(b_u_east, -b_u_west, atol=self.atol)
        else:
            npt.assert_allclose(b_u_east, b_u_west, atol=self.atol)

    def test_symmetry_when_flipping(self, sample_3d_grid, sample_prism):
        """
        Test symmetry of magnetic_e when flipping its direction
        """
        easting, northing, upward = sample_3d_grid
        # Keep only points with upward on top of the prism center
        _, _, prism_center_u = get_prism_center(sample_prism)
        is_top_or_equal = upward >= prism_center_u
        easting = easting[is_top_or_equal]
        northing = northing[is_top_or_equal]
        upward = upward[is_top_or_equal]
        # Define two magnetic moments
        magnetization_up = np.array([0, 0, 500.0])
        magnetization_down = np.array([0, 0, -500.0])
        # Compute the magnetic field generated by each moment on the
        # observation points
        b_u_up = np.array(
            list(
                magnetic_u(e, n, u, sample_prism, magnetization_up)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        b_u_down = np.array(
            list(
                magnetic_u(e, n, u, sample_prism, magnetization_down)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        # Check if the sign gets inverted
        npt.assert_allclose(b_u_up, -b_u_down)


class TestMagneticField:
    """
    Test magnetic_field against magnetic_easting, magnetic_northing and
    magnetic_upward

    Check if the components returned by magnetic_field match the individual
    ones computed by each one of the other functions.
    """

    def test_magnetic_field(self, sample_3d_grid, sample_prism, sample_magnetization):
        """
        Test magnetic_field against each one of the other functions
        """
        # Compute all components of B using magnetic_field
        b = np.array(
            list(
                magnetic_field(e, n, u, sample_prism, sample_magnetization)
                for e, n, u in zip(*sample_3d_grid)
            )
        )
        b_e, b_n, b_u = tuple(b[:, i] for i in range(3))
        # Computed the individual fields
        b_e_expected = np.array(
            list(
                magnetic_e(e, n, u, sample_prism, sample_magnetization)
                for e, n, u in zip(*sample_3d_grid)
            )
        )
        b_n_expected = np.array(
            list(
                magnetic_n(e, n, u, sample_prism, sample_magnetization)
                for e, n, u in zip(*sample_3d_grid)
            )
        )
        b_u_expected = np.array(
            list(
                magnetic_u(e, n, u, sample_prism, sample_magnetization)
                for e, n, u in zip(*sample_3d_grid)
            )
        )
        npt.assert_allclose(b_e, b_e_expected)
        npt.assert_allclose(b_n, b_n_expected)
        npt.assert_allclose(b_u, b_u_expected)


class TestDivergenceOfB:
    """
    Test if the divergence of the magnetic field is equal to zero

    Compute the derivatives of B through finite differences
    """

    # Displacement used in the finite differences
    delta = 1e-6

    def get_b_ee_finite_differences(self, coordinates, prism, magnetization):
        """
        Compute b_ee using finite differences
        """
        # Get original coordinates
        easting, northing, upward = coordinates
        # Shift coordinates using delta
        easting_shifted = easting + self.delta
        # Compute b_e on original and shifted coordinates
        b_e = np.array(
            list(
                magnetic_e(e, n, u, prism, magnetization)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        b_e_shifted = np.array(
            list(
                magnetic_e(e, n, u, prism, magnetization)
                for e, n, u in zip(easting_shifted, northing, upward)
            )
        )
        # Compute b_ee
        b_ee = (b_e_shifted - b_e) / self.delta
        return b_ee

    def get_b_nn_finite_differences(self, coordinates, prism, magnetization):
        """
        Compute b_nn using finite differences
        """
        # Get original coordinates
        easting, northing, upward = coordinates
        # Shift coordinates using delta
        northing_shifted = northing + self.delta
        # Compute b_e on original and shifted coordinates
        b_n = np.array(
            list(
                magnetic_n(e, n, u, prism, magnetization)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        b_n_shifted = np.array(
            list(
                magnetic_n(e, n, u, prism, magnetization)
                for e, n, u in zip(easting, northing_shifted, upward)
            )
        )
        # Compute b_nn
        b_nn = (b_n_shifted - b_n) / self.delta
        return b_nn

    def get_b_uu_finite_differences(self, coordinates, prism, magnetization):
        """
        Compute b_uu using finite differences
        """
        # Get original coordinates
        easting, northing, upward = coordinates
        # Shift coordinates using delta
        upward_shifted = upward + self.delta
        # Compute b_e on original and shifted coordinates
        b_u = np.array(
            list(
                magnetic_u(e, n, u, prism, magnetization)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        b_u_shifted = np.array(
            list(
                magnetic_u(e, n, u, prism, magnetization)
                for e, n, u in zip(easting, northing, upward_shifted)
            )
        )
        # Compute b_uu
        b_uu = (b_u_shifted - b_u) / self.delta
        return b_uu

    def test_divergence_of_b(self, sample_3d_grid, sample_prism, sample_magnetization):
        # Compute b_ee, b_nn and b_uu using finite differences
        b_ee = self.get_b_ee_finite_differences(
            sample_3d_grid, sample_prism, sample_magnetization
        )
        b_nn = self.get_b_nn_finite_differences(
            sample_3d_grid, sample_prism, sample_magnetization
        )
        b_uu = self.get_b_uu_finite_differences(
            sample_3d_grid, sample_prism, sample_magnetization
        )
        # Check if the divergence of B is zero
        npt.assert_allclose(-b_uu, b_ee + b_nn, atol=1e-11)
