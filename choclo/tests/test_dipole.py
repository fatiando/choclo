# Copyright (c) 2022 The Choclo Developers-.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test magnetic forward modelling functions for a dipole
"""
import pytest
import numpy as np
import numpy.testing as npt

from ..dipole import magnetic_e, magnetic_n, magnetic_u


@pytest.fixture(name="sample_dipole")
def fixture_sample_dipole():
    """
    Return the location of a sample dipole
    """
    return 40.5, 32.4, -15.3


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


class TestSymmetryBu:
    """
    Test symmetry of upward component of the magnetic field
    """

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
                magnetic_u(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_top.ravel()
                )
            )
        )
        b_u_bottom = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_bottom.ravel()
                )
            )
        )
        # Check symmetry between top and bottom
        atol = 1e-22  # absolute tolerance for values near zero
        if magnetic_moment[2] != 0:
            npt.assert_allclose(b_u_top, b_u_bottom, atol=atol)
        else:
            npt.assert_allclose(b_u_top, -b_u_bottom, atol=atol)

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
                magnetic_u(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing_north.ravel(), upward.ravel()
                )
            )
        )
        b_u_south = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing_south.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between south and north
        atol = 1e-22  # absolute tolerance for values near zero
        if magnetic_moment[1] != 0:
            npt.assert_allclose(b_u_north, -b_u_south, atol=atol)
        else:
            npt.assert_allclose(b_u_north, b_u_south, atol=atol)

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
                magnetic_u(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting_east.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        b_u_west = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting_west.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between east and west
        atol = 1e-22  # absolute tolerance for values near zero
        if magnetic_moment[0] != 0:
            npt.assert_allclose(b_u_east, -b_u_west, atol=atol)
        else:
            npt.assert_allclose(b_u_east, b_u_west, atol=atol)

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
                magnetic_u(e, n, u, *sample_dipole, magnetic_moment_up)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        b_u_down = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, magnetic_moment_down)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        # Check if the sign gets inverted
        npt.assert_allclose(b_u_up, -b_u_down)


class TestSymmetryBe:
    """
    Test symmetry of easting component of the magnetic field
    """

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
                magnetic_e(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_top.ravel()
                )
            )
        )
        b_e_bottom = np.array(
            list(
                magnetic_e(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_bottom.ravel()
                )
            )
        )
        # Check symmetry between top and bottom
        atol = 1e-22  # absolute tolerance for values near zero
        if magnetic_moment[2] != 0:
            npt.assert_allclose(b_e_top, -b_e_bottom, atol=atol)
        else:
            npt.assert_allclose(b_e_top, b_e_bottom, atol=atol)

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
                magnetic_e(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing_north.ravel(), upward.ravel()
                )
            )
        )
        b_e_south = np.array(
            list(
                magnetic_e(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing_south.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between south and north
        atol = 1e-22  # absolute tolerance for values near zero
        if magnetic_moment[1] != 0:
            npt.assert_allclose(b_e_north, -b_e_south, atol=atol)
        else:
            npt.assert_allclose(b_e_north, b_e_south, atol=atol)

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
                magnetic_e(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting_east.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        b_e_west = np.array(
            list(
                magnetic_e(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting_west.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between west and east
        atol = 1e-22  # absolute tolerance for values near zero
        if magnetic_moment[0] != 0:
            npt.assert_allclose(b_e_east, b_e_west, atol=atol)
        else:
            npt.assert_allclose(b_e_east, -b_e_west, atol=atol)

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
                magnetic_e(e, n, u, *sample_dipole, magnetic_moment_east)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        b_e_west = np.array(
            list(
                magnetic_e(e, n, u, *sample_dipole, magnetic_moment_west)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        # Check if the sign gets inverted
        npt.assert_allclose(b_e_east, -b_e_west)


class TestSymmetryBn:
    """
    Test symmetry of easting component of the magnetic field
    """

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
                magnetic_n(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_top.ravel()
                )
            )
        )
        b_n_bottom = np.array(
            list(
                magnetic_n(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing.ravel(), upward_bottom.ravel()
                )
            )
        )
        # Check symmetry between top and bottom
        atol = 1e-22  # absolute tolerance for values near zero
        if magnetic_moment[2] != 0:
            npt.assert_allclose(b_n_top, -b_n_bottom, atol=atol)
        else:
            npt.assert_allclose(b_n_top, b_n_bottom, atol=atol)

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
                magnetic_n(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing_north.ravel(), upward.ravel()
                )
            )
        )
        b_n_south = np.array(
            list(
                magnetic_n(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting.ravel(), northing_south.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between south and north
        atol = 1e-22  # absolute tolerance for values near zero
        if magnetic_moment[1] != 0:
            npt.assert_allclose(b_n_north, b_n_south, atol=atol)
        else:
            npt.assert_allclose(b_n_north, -b_n_south, atol=atol)

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
                magnetic_n(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting_east.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        b_n_west = np.array(
            list(
                magnetic_n(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(
                    easting_west.ravel(), northing.ravel(), upward.ravel()
                )
            )
        )
        # Check symmetry between west and east
        atol = 1e-22  # absolute tolerance for values near zero
        if magnetic_moment[0] != 0:
            npt.assert_allclose(b_n_east, -b_n_west, atol=atol)
        else:
            npt.assert_allclose(b_n_east, b_n_west, atol=atol)

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
                magnetic_n(e, n, u, *sample_dipole, magnetic_moment_north)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        b_n_south = np.array(
            list(
                magnetic_n(e, n, u, *sample_dipole, magnetic_moment_south)
                for e, n, u in zip(easting, northing, upward)
            )
        )
        # Check if the sign gets inverted
        npt.assert_allclose(b_n_north, -b_n_south)
