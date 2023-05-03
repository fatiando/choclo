# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test magnetic forward modelling functions for a dipole
"""
import numpy as np
import numpy.testing as npt
import pytest

from ..dipole import magnetic_e, magnetic_field, magnetic_n, magnetic_u


@pytest.fixture(name="sample_dipole")
def fixture_sample_dipole():
    """
    Return the location of a sample dipole
    """
    return 40.5, 32.4, -15.3


@pytest.fixture(name="sample_magnetic_moment")
def fixture_sample_magnetic_moment():
    """
    Return a sample magnetic moment
    """
    return np.array([780.3, -230.4, 1030])


@pytest.fixture(name="sample_3d_grid")
def fixture_sample_3d_grid(sample_dipole):
    """
    Return a set of observation points in a 3d grid

    These observation points are centered around the dipole, but they don't
    include its position.
    """
    easting = np.linspace(-40, 40, 11)
    northing = np.linspace(-40, 40, 11)
    upward = np.linspace(-40, 40, 11)
    # Compute meshgrid
    easting, northing, upward = np.meshgrid(easting, northing, upward)
    # Remove the location of the dipole
    is_dipole = (easting == 0) & (northing == 0) & (upward == 0)
    easting = easting[np.logical_not(is_dipole)]
    northing = northing[np.logical_not(is_dipole)]
    upward = upward[np.logical_not(is_dipole)]
    # Shift coordinates
    easting += sample_dipole[0]
    northing += sample_dipole[1]
    upward += sample_dipole[2]
    return easting, northing, upward


class TestSymmetryBe:
    """
    Test symmetry of easting component of the magnetic field
    """

    atol = 1e-22  # absolute tolerance for values near zero

    @pytest.mark.parametrize(
        "magnetic_moment",
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
        self, sample_3d_grid, sample_dipole, magnetic_moment
    ):
        """
        Test symmetry of magnetic_e across the easting-northing plane that
        passes through the location of the dipole
        """
        easting, northing, upward = sample_3d_grid
        # Keep only the observation points that are above the dipole
        is_top = upward > sample_dipole[2]
        easting = easting[is_top]
        northing = northing[is_top]
        upward_top = upward[is_top]
        # Create a symmetrical upward coordinate for points below the dipole
        upward_bottom = 2 * sample_dipole[2] - upward_top
        # Compute magnetic_e on every observation point
        magnetic_moment = np.array(magnetic_moment)
        b_e_top = np.array(
            list(
                magnetic_e(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_top.ravel()
                )
            )
        )
        b_e_bottom = np.array(
            list(
                magnetic_e(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_bottom.ravel()
                )
            )
        )
        # Check symmetry between top and bottom
        if magnetic_moment[2] != 0:
            npt.assert_allclose(b_e_top, -b_e_bottom, atol=self.atol)
        else:
            npt.assert_allclose(b_e_top, b_e_bottom, atol=self.atol)

    @pytest.mark.parametrize(
        "magnetic_moment",
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
        self, sample_3d_grid, sample_dipole, magnetic_moment
    ):
        """
        Test symmetry of magnetic_e across the easting-upward plane that
        passes through the location of the dipole
        """
        easting, northing, upward = sample_3d_grid
        # Keep only the observation points that are north the dipole
        is_north = northing > sample_dipole[1]
        easting = easting[is_north]
        northing_north = northing[is_north]
        upward = upward[is_north]
        # Create a symmetrical upward coordinate for points south the dipole
        northing_south = 2 * sample_dipole[1] - northing_north
        # Compute magnetic_e on every observation point
        magnetic_moment = np.array(magnetic_moment)
        b_e_north = np.array(
            list(
                magnetic_e(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing_north.ravel(), upward.ravel()
                )
            )
        )
        b_e_south = np.array(
            list(
                magnetic_e(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing_south.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between south and north
        if magnetic_moment[1] != 0:
            npt.assert_allclose(b_e_north, -b_e_south, atol=self.atol)
        else:
            npt.assert_allclose(b_e_north, b_e_south, atol=self.atol)

    @pytest.mark.parametrize(
        "magnetic_moment",
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
        self, sample_3d_grid, sample_dipole, magnetic_moment
    ):
        """
        Test symmetry of magnetic_e across the northing-upward plane that
        passes through the location of the dipole
        """
        easting, northing, upward = sample_3d_grid
        # Keep only the observation points that are east the dipole
        is_east = easting > sample_dipole[0]
        easting_east = easting[is_east]
        northing = northing[is_east]
        upward = upward[is_east]
        # Create a symmetrical upward coordinate for points west the dipole
        easting_west = 2 * sample_dipole[0] - easting_east
        # Compute magnetic_e on every observation point
        magnetic_moment = np.array(magnetic_moment)
        b_e_east = np.array(
            list(
                magnetic_e(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting_east.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        b_e_west = np.array(
            list(
                magnetic_e(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting_west.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between west and east
        if magnetic_moment[0] != 0:
            npt.assert_allclose(b_e_east, b_e_west, atol=self.atol)
        else:
            npt.assert_allclose(b_e_east, -b_e_west, atol=self.atol)

    def test_symmetry_when_flipping(self, sample_3d_grid, sample_dipole):
        """
        Test symmetry of magnetic_e when flipping its direction
        """
        easting, northing, upward = sample_3d_grid
        # Keep only points with easting on east of the dipole
        is_east_or_equal = easting >= sample_dipole[0]
        easting = easting[is_east_or_equal]
        northing = northing[is_east_or_equal]
        upward = upward[is_east_or_equal]
        # Define two magnetic moments
        magnetic_moment_east = np.array([500.0, 0, 0])
        magnetic_moment_west = np.array([-500.0, 0, 0])
        # Compute the magnetic field generated by each moment on the
        # observation points
        b_e_east = np.array(
            list(
                magnetic_e(e, n, u, *sample_dipole, *magnetic_moment_east)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        b_e_west = np.array(
            list(
                magnetic_e(e, n, u, *sample_dipole, *magnetic_moment_west)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        # Check if the sign gets inverted
        npt.assert_allclose(b_e_east, -b_e_west)


class TestSymmetryBn:
    """
    Test symmetry of easting component of the magnetic field
    """

    atol = 1e-22  # absolute tolerance for values near zero

    @pytest.mark.parametrize(
        "magnetic_moment",
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
        self, sample_3d_grid, sample_dipole, magnetic_moment
    ):
        """
        Test symmetry of magnetic_n across the easting-northing plane that
        passes through the location of the dipole
        """
        easting, northing, upward = sample_3d_grid
        # Keep only the observation points that are above the dipole
        is_top = upward > sample_dipole[2]
        easting = easting[is_top]
        northing = northing[is_top]
        upward_top = upward[is_top]
        # Create a symmetrical upward coordinate for points below the dipole
        upward_bottom = 2 * sample_dipole[2] - upward_top
        # Compute magnetic_n on every observation point
        magnetic_moment = np.array(magnetic_moment)
        b_n_top = np.array(
            list(
                magnetic_n(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_top.ravel()
                )
            )
        )
        b_n_bottom = np.array(
            list(
                magnetic_n(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_bottom.ravel()
                )
            )
        )
        # Check symmetry between top and bottom
        if magnetic_moment[2] != 0:
            npt.assert_allclose(b_n_top, -b_n_bottom, atol=self.atol)
        else:
            npt.assert_allclose(b_n_top, b_n_bottom, atol=self.atol)

    @pytest.mark.parametrize(
        "magnetic_moment",
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
        self, sample_3d_grid, sample_dipole, magnetic_moment
    ):
        """
        Test symmetry of magnetic_n across the easting-upward plane that
        passes through the location of the dipole
        """
        easting, northing, upward = sample_3d_grid
        # Keep only the observation points that are north the dipole
        is_north = northing > sample_dipole[1]
        easting = easting[is_north]
        northing_north = northing[is_north]
        upward = upward[is_north]
        # Create a symmetrical upward coordinate for points south the dipole
        northing_south = 2 * sample_dipole[1] - northing_north
        # Compute magnetic_n on every observation point
        magnetic_moment = np.array(magnetic_moment)
        b_n_north = np.array(
            list(
                magnetic_n(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing_north.ravel(), upward.ravel()
                )
            )
        )
        b_n_south = np.array(
            list(
                magnetic_n(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing_south.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between south and north
        if magnetic_moment[1] != 0:
            npt.assert_allclose(b_n_north, b_n_south, atol=self.atol)
        else:
            npt.assert_allclose(b_n_north, -b_n_south, atol=self.atol)

    @pytest.mark.parametrize(
        "magnetic_moment",
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
        self, sample_3d_grid, sample_dipole, magnetic_moment
    ):
        """
        Test symmetry of magnetic_n across the northing-upward plane that
        passes through the location of the dipole
        """
        easting, northing, upward = sample_3d_grid
        # Keep only the observation points that are east the dipole
        is_east = easting > sample_dipole[0]
        easting_east = easting[is_east]
        northing = northing[is_east]
        upward = upward[is_east]
        # Create a symmetrical upward coordinate for points west the dipole
        easting_west = 2 * sample_dipole[0] - easting_east
        # Compute magnetic_n on every observation point
        magnetic_moment = np.array(magnetic_moment)
        b_n_east = np.array(
            list(
                magnetic_n(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting_east.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        b_n_west = np.array(
            list(
                magnetic_n(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting_west.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between west and east
        if magnetic_moment[0] != 0:
            npt.assert_allclose(b_n_east, -b_n_west, atol=self.atol)
        else:
            npt.assert_allclose(b_n_east, b_n_west, atol=self.atol)

    def test_symmetry_when_flipping(self, sample_3d_grid, sample_dipole):
        """
        Test symmetry of magnetic_n when flipping its direction
        """
        easting, northing, upward = sample_3d_grid
        # Keep only points with northing on north of the dipole
        is_north_or_equal = northing >= sample_dipole[1]
        easting = easting[is_north_or_equal]
        northing = northing[is_north_or_equal]
        upward = upward[is_north_or_equal]
        # Define two magnetic moments
        magnetic_moment_north = np.array([0, 500.0, 0])
        magnetic_moment_south = np.array([0, -500.0, 0])
        # Compute the magnetic field generated by each moment on the
        # observation points
        b_n_north = np.array(
            list(
                magnetic_n(e, n, u, *sample_dipole, *magnetic_moment_north)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        b_n_south = np.array(
            list(
                magnetic_n(e, n, u, *sample_dipole, *magnetic_moment_south)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        # Check if the sign gets inverted
        npt.assert_allclose(b_n_north, -b_n_south)


class TestSymmetryBu:
    """
    Test symmetry of upward component of the magnetic field
    """

    atol = 1e-22  # absolute tolerance for values near zero

    @pytest.mark.parametrize(
        "magnetic_moment",
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
        self, sample_3d_grid, sample_dipole, magnetic_moment
    ):
        """
        Test symmetry of magnetic_e across the easting-northing plane that
        passes through the location of the dipole
        """
        easting, northing, upward = sample_3d_grid
        # Keep only the observation points that are above the dipole
        is_top = upward > sample_dipole[2]
        easting = easting[is_top]
        northing = northing[is_top]
        upward_top = upward[is_top]
        # Create a symmetrical upward coordinate for points below the dipole
        upward_bottom = 2 * sample_dipole[2] - upward_top
        # Compute magnetic_u on every observation point
        magnetic_moment = np.array(magnetic_moment)
        b_u_top = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_top.ravel()
                )
            )
        )
        b_u_bottom = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_bottom.ravel()
                )
            )
        )
        # Check symmetry between top and bottom
        if magnetic_moment[2] != 0:
            npt.assert_allclose(b_u_top, b_u_bottom, atol=self.atol)
        else:
            npt.assert_allclose(b_u_top, -b_u_bottom, atol=self.atol)

    @pytest.mark.parametrize(
        "magnetic_moment",
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
        self, sample_3d_grid, sample_dipole, magnetic_moment
    ):
        """
        Test symmetry of magnetic_e across the easting-upward plane that
        passes through the location of the dipole
        """
        easting, northing, upward = sample_3d_grid
        # Keep only the observation points that are north the dipole
        is_north = northing > sample_dipole[1]
        easting = easting[is_north]
        northing_north = northing[is_north]
        upward = upward[is_north]
        # Create a symmetrical upward coordinate for points south the dipole
        northing_south = 2 * sample_dipole[1] - northing_north
        # Compute magnetic_u on every observation point
        magnetic_moment = np.array(magnetic_moment)
        b_u_north = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing_north.ravel(), upward.ravel()
                )
            )
        )
        b_u_south = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing_south.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between south and north
        if magnetic_moment[1] != 0:
            npt.assert_allclose(b_u_north, -b_u_south, atol=self.atol)
        else:
            npt.assert_allclose(b_u_north, b_u_south, atol=self.atol)

    @pytest.mark.parametrize(
        "magnetic_moment",
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
        self, sample_3d_grid, sample_dipole, magnetic_moment
    ):
        """
        Test symmetry of magnetic_e across the northing-upward plane that
        passes through the location of the dipole
        """
        easting, northing, upward = sample_3d_grid
        # Keep only the observation points that are east the dipole
        is_east = easting > sample_dipole[0]
        easting_east = easting[is_east]
        northing = northing[is_east]
        upward = upward[is_east]
        # Create a symmetrical upward coordinate for points west the dipole
        easting_west = 2 * sample_dipole[0] - easting_east
        # Compute magnetic_u on every observation point
        magnetic_moment = np.array(magnetic_moment)
        b_u_east = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting_east.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        b_u_west = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, *magnetic_moment)
                for e, n, u in zip(
                    easting_west.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between east and west
        if magnetic_moment[0] != 0:
            npt.assert_allclose(b_u_east, -b_u_west, atol=self.atol)
        else:
            npt.assert_allclose(b_u_east, b_u_west, atol=self.atol)

    def test_symmetry_when_flipping(self, sample_3d_grid, sample_dipole):
        """
        Test symmetry of magnetic_e when flipping its direction
        """
        easting, northing, upward = sample_3d_grid
        # Keep only points with upward on top of the dipole
        is_top_or_equal = upward >= sample_dipole[2]
        easting = easting[is_top_or_equal]
        northing = northing[is_top_or_equal]
        upward = upward[is_top_or_equal]
        # Define two magnetic moments
        magnetic_moment_up = np.array([0, 0, 500.0])
        magnetic_moment_down = np.array([0, 0, -500.0])
        # Compute the magnetic field generated by each moment on the
        # observation points
        b_u_up = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, *magnetic_moment_up)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        b_u_down = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, *magnetic_moment_down)
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

    def test_magnetic_field(
        self, sample_3d_grid, sample_dipole, sample_magnetic_moment
    ):
        """
        Test magnetic_field against each one of the other functions
        """
        # Compute all components of B using magnetic_field
        b = np.array(
            list(
                magnetic_field(e, n, u, *sample_dipole, *sample_magnetic_moment)
                for e, n, u in zip(*sample_3d_grid)
            )
        )
        b_e, b_n, b_u = tuple(b[:, i] for i in range(3))
        # Computed the individual fields
        b_e_expected = np.array(
            list(
                magnetic_e(e, n, u, *sample_dipole, *sample_magnetic_moment)
                for e, n, u in zip(*sample_3d_grid)
            )
        )
        b_n_expected = np.array(
            list(
                magnetic_n(e, n, u, *sample_dipole, *sample_magnetic_moment)
                for e, n, u in zip(*sample_3d_grid)
            )
        )
        b_u_expected = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, *sample_magnetic_moment)
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
    delta = 1e-5

    def get_b_ee_finite_differences(self, coordinates, dipole, magnetic_moment):
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
                magnetic_e(e, n, u, *dipole, *magnetic_moment)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        b_e_shifted = np.array(
            list(
                magnetic_e(e, n, u, *dipole, *magnetic_moment)
                for e, n, u in zip(easting_shifted, northing, upward)
            )
        )
        # Compute b_ee
        b_ee = (b_e_shifted - b_e) / self.delta
        return b_ee

    def get_b_nn_finite_differences(self, coordinates, dipole, magnetic_moment):
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
                magnetic_n(e, n, u, *dipole, *magnetic_moment)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        b_n_shifted = np.array(
            list(
                magnetic_n(e, n, u, *dipole, *magnetic_moment)
                for e, n, u in zip(easting, northing_shifted, upward)
            )
        )
        # Compute b_nn
        b_nn = (b_n_shifted - b_n) / self.delta
        return b_nn

    def get_b_uu_finite_differences(self, coordinates, dipole, magnetic_moment):
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
                magnetic_u(e, n, u, *dipole, *magnetic_moment)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        b_u_shifted = np.array(
            list(
                magnetic_u(e, n, u, *dipole, *magnetic_moment)
                for e, n, u in zip(easting, northing, upward_shifted)
            )
        )
        # Compute b_uu
        b_uu = (b_u_shifted - b_u) / self.delta
        return b_uu

    def test_divergence_of_b(
        self, sample_3d_grid, sample_dipole, sample_magnetic_moment
    ):
        # Compute b_ee, b_nn and b_uu using finite differences
        b_ee = self.get_b_ee_finite_differences(
            sample_3d_grid, sample_dipole, sample_magnetic_moment
        )
        b_nn = self.get_b_nn_finite_differences(
            sample_3d_grid, sample_dipole, sample_magnetic_moment
        )
        b_uu = self.get_b_uu_finite_differences(
            sample_3d_grid, sample_dipole, sample_magnetic_moment
        )
        # Check if the divergence of B is zero
        npt.assert_allclose(-b_uu, b_ee + b_nn, atol=1e-12)
