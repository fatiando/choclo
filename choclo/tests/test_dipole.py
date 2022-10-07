# Copyright (c) 2022 The Choclo Developers.
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

from ..dipole import magnetic_u


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
    # Ravel the arrays
    easting = easting.ravel()
    northing = northing.ravel()
    upward = upward.ravel()
    return easting, northing, upward


class TestSymmetryBu:
    """
    Test symmetry of upward component of the magnetic field
    """

    @pytest.mark.parametrize("magnetic_moment", [(0, 0, 500), (0, 0, -500)])
    def test_symmetry_across_easting_northing(
        self, sample_3d_grid, sample_dipole, magnetic_moment
    ):
        """
        Test symmetry of magnetic_e across the easting-northing plane that
        passes through the location of the dipole
        """
        easting, northing, upward = sample_3d_grid
        # Split the observation points between top and bottom
        is_top = upward > sample_dipole[2]
        easting_top = easting[is_top]
        northing_top = northing[is_top]
        upward_top = upward[is_top]
        # Reverse the order of the bottoms to match the order of the top points
        is_bottom = upward < sample_dipole[2]
        easting_bottom = easting[is_bottom][::-1]
        northing_bottom = northing[is_bottom][::-1]
        upward_bottom = upward[is_bottom][::-1]
        # Compute magnetic_u on every observation point
        magnetic_moment = np.array(magnetic_moment)
        b_u_top = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(easting_top, northing_top, upward_top)
            )
        )
        b_u_bottom = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(easting_bottom, northing_bottom, upward_bottom)
            )
        )
        # Check symmetry between top and bottom
        atol = 1e-22  # absolute tolerance for values near zero
        npt.assert_allclose(b_u_top, b_u_bottom, atol=atol)

    @pytest.mark.parametrize("magnetic_moment", [(0, 0, 500), (0, 0, -500)])
    def test_symmetry_across_easting_upward(
        self, sample_3d_grid, sample_dipole, magnetic_moment
    ):
        """
        Test symmetry of magnetic_e across the easting-upward plane that
        passes through the location of the dipole
        """
        easting, northing, upward = sample_3d_grid
        # Split the observation points between south and north
        is_north = northing > sample_dipole[1]
        easting_north = easting[is_north]
        northing_north = northing[is_north]
        upward_north = upward[is_north]
        # Reverse the order of the south points to match the order of the
        # north ones
        is_south = northing < sample_dipole[1]
        easting_south = easting[is_south][::-1]
        northing_south = northing[is_south][::-1]
        upward_south = upward[is_south][::-1]
        # Compute magnetic_u on every observation point
        magnetic_moment = np.array(magnetic_moment)
        b_u_north = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(easting_north, northing_north, upward_north)
            )
        )
        b_u_south = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(easting_south, northing_south, upward_south)
            )
        )
        # Check symmetry between top and bottom
        atol = 1e-22  # absolute tolerance for values near zero
        npt.assert_allclose(b_u_north, b_u_south, atol=atol)

    @pytest.mark.parametrize("magnetic_moment", [(0, 0, 500), (0, 0, -500)])
    def test_symmetry_across_northing_upward(
        self, sample_3d_grid, sample_dipole, magnetic_moment
    ):
        """
        Test symmetry of magnetic_e across the northing-upward plane that
        passes through the location of the dipole
        """
        easting, northing, upward = sample_3d_grid
        # Split the observation points between west and east
        is_east = easting > sample_dipole[0]
        easting_east = easting[is_east]
        northing_east = northing[is_east]
        upward_east = upward[is_east]
        # Reverse the order of the south points to match the order of the
        # north ones
        is_west = easting < sample_dipole[0]
        easting_west = easting[is_west][::-1]
        northing_west = northing[is_west][::-1]
        upward_west = upward[is_west][::-1]
        # Compute magnetic_u on every observation point
        magnetic_moment = np.array(magnetic_moment)
        b_u_east = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(easting_east, northing_east, upward_east)
            )
        )
        b_u_west = np.array(
            list(
                magnetic_u(e, n, u, *sample_dipole, magnetic_moment)
                for e, n, u in zip(easting_west, northing_west, upward_west)
            )
        )
        # Check symmetry between top and bottom
        atol = 1e-22  # absolute tolerance for values near zero
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
