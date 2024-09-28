# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test magnetic forward modelling functions for rectangular prisms
"""
import itertools
import numpy as np
import numpy.testing as npt
import pytest

from ..prism import (
    magnetic_e,
    magnetic_ee,
    magnetic_en,
    magnetic_eu,
    magnetic_field,
    magnetic_n,
    magnetic_nn,
    magnetic_nu,
    magnetic_u,
    magnetic_uu,
)

COMPONENTS = {
    "e": magnetic_e,
    "n": magnetic_n,
    "u": magnetic_u,
    "ee": magnetic_ee,
    "nn": magnetic_nn,
    "uu": magnetic_uu,
    "en": magnetic_en,
    "eu": magnetic_eu,
    "nu": magnetic_nu,
}


def get_forward_function(component: str):
    """Return desired magnetic forward function."""
    component = "".join(sorted(component))
    return COMPONENTS[component]


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


def evaluate(forward_func, coordinates, prism, magnetization):
    """
    Evaluate a forward function on a set of observation points.

    Parameters
    ----------
    forward_func : callable
        Forward modelling function to evaluate.
    coordinates : tuple of (n_data) arrays
        Coordinates of the observation points.
    prism : (6) array
        Boundaries of the prism.
    magnetization : (3) array
        Magnetization vector of the prism.

    Returns
    -------
    array
        Array with the result of evaluating the ``forward_func`` on every
        observation point.
    """
    coordinates = tuple(c.ravel() for c in coordinates)
    result = np.array(
        list(
            forward_func(e, n, u, *prism, *magnetization)
            for e, n, u in zip(*coordinates)
        )
    )
    return result


def finite_differences(
    coordinates, prism, magnetization, direction, forward_func, delta=1e-6
):
    """
    Compute spatial derivatives through finite differences.

    Parameters
    ----------
    coordinates : tuple of (n_data) arrays
        Coordinates of the observation points.
    prism : (6) array
        Boundaries of the prism.
    magnetization : (3) array
        Magnetization vector of the prism.
    direction : {"e", "n", "u"}
        Direction along which take the derivative.
    forward_func : callable
        Forward modelling function to use to compute the finite difference.
    delta : float, optional
        Displacement for the finite differences.

    Returns
    -------
    (n_data) array
        Array with the derivatives of the ``forward_func`` on each observation
        point calculated using finite differences.
    """
    # Get original coordinates
    easting, northing, upward = coordinates
    if direction == "e":
        shifted_coords = (easting + delta, northing, upward)
    elif direction == "n":
        shifted_coords = (easting, northing + delta, upward)
    elif direction == "u":
        shifted_coords = (easting, northing, upward + delta)
    else:
        ValueError(f"Invalid direction '{direction}'")
    # Compute field on original and shifted coordinates
    field = evaluate(forward_func, coordinates, prism, magnetization)
    field_shifted = evaluate(forward_func, shifted_coords, prism, magnetization)
    # Compute spatial derivative
    spatial_derivative = (field_shifted - field) / delta
    return spatial_derivative


class TestSymmetryBe:
    """
    Test symmetry of easting component of the magnetic field
    """

    atol = 1e-17  # absolute tolerance for values near zero
    MAGNETIZATIONS = [
        (500, 0, 0),
        (-500, 0, 0),
        (0, 500, 0),
        (0, -500, 0),
        (0, 0, 500),
        (0, 0, -500),
    ]

    @pytest.mark.parametrize("magnetization", MAGNETIZATIONS)
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
        b_e_top = evaluate(
            magnetic_e, (easting, northing, upward_top), sample_prism, magnetization
        )
        b_e_bottom = evaluate(
            magnetic_e, (easting, northing, upward_bottom), sample_prism, magnetization
        )
        # Check symmetry between top and bottom
        if magnetization[2] != 0:
            npt.assert_allclose(b_e_top, -b_e_bottom, atol=self.atol)
        else:
            npt.assert_allclose(b_e_top, b_e_bottom, atol=self.atol)

    @pytest.mark.parametrize("magnetization", MAGNETIZATIONS)
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
        b_e_north = evaluate(
            magnetic_e, (easting, northing_north, upward), sample_prism, magnetization
        )
        b_e_south = evaluate(
            magnetic_e, (easting, northing_south, upward), sample_prism, magnetization
        )
        # Check symmetry between south and north
        if magnetization[1] != 0:
            npt.assert_allclose(b_e_north, -b_e_south, atol=self.atol)
        else:
            npt.assert_allclose(b_e_north, b_e_south, atol=self.atol)

    @pytest.mark.parametrize("magnetization", MAGNETIZATIONS)
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
        b_e_east = evaluate(
            magnetic_e, (easting_east, northing, upward), sample_prism, magnetization
        )
        b_e_west = evaluate(
            magnetic_e, (easting_west, northing, upward), sample_prism, magnetization
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
        b_e_east = evaluate(
            magnetic_e, (easting, northing, upward), sample_prism, magnetization_east
        )
        b_e_west = evaluate(
            magnetic_e, (easting, northing, upward), sample_prism, magnetization_west
        )
        # Check if the sign gets inverted
        npt.assert_allclose(b_e_east, -b_e_west)


class TestSymmetryBn:
    """
    Test symmetry of easting component of the magnetic field
    """

    atol = 1e-17  # absolute tolerance for values near zero
    MAGNETIZATIONS = [
        (500, 0, 0),
        (-500, 0, 0),
        (0, 500, 0),
        (0, -500, 0),
        (0, 0, 500),
        (0, 0, -500),
    ]

    @pytest.mark.parametrize("magnetization", MAGNETIZATIONS)
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
        b_n_top = evaluate(
            magnetic_n, (easting, northing, upward_top), sample_prism, magnetization
        )
        b_n_bottom = evaluate(
            magnetic_n, (easting, northing, upward_bottom), sample_prism, magnetization
        )
        # Check symmetry between top and bottom
        if magnetization[2] != 0:
            npt.assert_allclose(b_n_top, -b_n_bottom, atol=self.atol)
        else:
            npt.assert_allclose(b_n_top, b_n_bottom, atol=self.atol)

    @pytest.mark.parametrize("magnetization", MAGNETIZATIONS)
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
        b_n_north = evaluate(
            magnetic_n, (easting, northing_north, upward), sample_prism, magnetization
        )
        b_n_south = evaluate(
            magnetic_n, (easting, northing_south, upward), sample_prism, magnetization
        )
        # Check symmetry between south and north
        if magnetization[1] != 0:
            npt.assert_allclose(b_n_north, b_n_south, atol=self.atol)
        else:
            npt.assert_allclose(b_n_north, -b_n_south, atol=self.atol)

    @pytest.mark.parametrize("magnetization", MAGNETIZATIONS)
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
        b_n_east = evaluate(
            magnetic_n, (easting_east, northing, upward), sample_prism, magnetization
        )
        b_n_west = evaluate(
            magnetic_n, (easting_west, northing, upward), sample_prism, magnetization
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
        b_n_north = evaluate(
            magnetic_n, (easting, northing, upward), sample_prism, magnetization_north
        )
        b_n_south = evaluate(
            magnetic_n, (easting, northing, upward), sample_prism, magnetization_south
        )
        # Check if the sign gets inverted
        npt.assert_allclose(b_n_north, -b_n_south)


class TestSymmetryBu:
    """
    Test symmetry of upward component of the magnetic field
    """

    atol = 1e-17  # absolute tolerance for values near zero
    MAGNETIZATIONS = [
        (500, 0, 0),
        (-500, 0, 0),
        (0, 500, 0),
        (0, -500, 0),
        (0, 0, 500),
        (0, 0, -500),
    ]

    @pytest.mark.parametrize("magnetization", MAGNETIZATIONS)
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
        b_u_top = evaluate(
            magnetic_u, (easting, northing, upward_top), sample_prism, magnetization
        )
        b_u_bottom = evaluate(
            magnetic_u, (easting, northing, upward_bottom), sample_prism, magnetization
        )
        # Check symmetry between top and bottom
        if magnetization[2] != 0:
            npt.assert_allclose(b_u_top, b_u_bottom, atol=self.atol)
        else:
            npt.assert_allclose(b_u_top, -b_u_bottom, atol=self.atol)

    @pytest.mark.parametrize("magnetization", MAGNETIZATIONS)
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
        b_u_north = evaluate(
            magnetic_u, (easting, northing_north, upward), sample_prism, magnetization
        )
        b_u_south = evaluate(
            magnetic_u, (easting, northing_south, upward), sample_prism, magnetization
        )
        # Check symmetry between south and north
        if magnetization[1] != 0:
            npt.assert_allclose(b_u_north, -b_u_south, atol=self.atol)
        else:
            npt.assert_allclose(b_u_north, b_u_south, atol=self.atol)

    @pytest.mark.parametrize("magnetization", MAGNETIZATIONS)
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
        b_u_east = evaluate(
            magnetic_u, (easting_east, northing, upward), sample_prism, magnetization
        )
        b_u_west = evaluate(
            magnetic_u, (easting_west, northing, upward), sample_prism, magnetization
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
        b_u_up = evaluate(
            magnetic_u, (easting, northing, upward), sample_prism, magnetization_up
        )
        b_u_down = evaluate(
            magnetic_u, (easting, northing, upward), sample_prism, magnetization_down
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
        b = evaluate(magnetic_field, sample_3d_grid, sample_prism, sample_magnetization)
        b_e, b_n, b_u = tuple(b[:, i] for i in range(3))
        # Computed the individual fields
        b_e_expected = evaluate(
            magnetic_e, sample_3d_grid, sample_prism, sample_magnetization
        )
        b_n_expected = evaluate(
            magnetic_n, sample_3d_grid, sample_prism, sample_magnetization
        )
        b_u_expected = evaluate(
            magnetic_u, sample_3d_grid, sample_prism, sample_magnetization
        )
        npt.assert_allclose(b_e, b_e_expected)
        npt.assert_allclose(b_n, b_n_expected)
        npt.assert_allclose(b_u, b_u_expected)


class TestDivergenceOfB:
    """
    Test if the divergence of the magnetic field is equal to zero.
    """

    def test_divergence_of_b(self, sample_3d_grid, sample_prism, sample_magnetization):
        """
        Test div of B through magnetic gradiometry analytical functions.
        """
        kwargs = dict(
            coordinates=sample_3d_grid,
            prism=sample_prism,
            magnetization=sample_magnetization,
        )
        b_ee = evaluate(magnetic_ee, **kwargs)
        b_nn = evaluate(magnetic_nn, **kwargs)
        b_uu = evaluate(magnetic_uu, **kwargs)
        # Check if the divergence of B is zero
        npt.assert_allclose(-b_uu, b_ee + b_nn, atol=1e-11)

    def test_divergence_of_b_finite_differences(
        self, sample_3d_grid, sample_prism, sample_magnetization
    ):
        """
        Test div of B computing its derivatives through finite differences.
        """
        delta = 1e-6
        kwargs = dict(
            coordinates=sample_3d_grid,
            prism=sample_prism,
            magnetization=sample_magnetization,
            delta=delta,
        )
        b_ee = finite_differences(direction="e", forward_func=magnetic_e, **kwargs)
        b_nn = finite_differences(direction="n", forward_func=magnetic_n, **kwargs)
        b_uu = finite_differences(direction="u", forward_func=magnetic_u, **kwargs)
        # Check if the divergence of B is zero
        npt.assert_allclose(-b_uu, b_ee + b_nn, atol=1e-11)


class TestMagneticFieldSingularities:
    """
    Test if magnetic field components behave as expected on their singular
    points

    Magnetic field components have singular points on:
    * prism vertices,
    * prism edges,
    * prism interior, and
    * prism faces normal to the magnetic component direction.

    For the first three cases, the forward modelling function should return
    ``np.nan``. For the last case, it should return the limit of the field when
    we approach from outside of the prism.
    """

    @pytest.fixture()
    def sample_prism(self):
        """
        Return the boundaries of the sample prism
        """
        west, east, south, north, bottom, top = -5.4, 10.1, 43.2, 79.5, -53.7, -44.3
        return np.array([west, east, south, north, bottom, top])

    def get_vertices(self, prism):
        """
        Return the vertices of the prism
        """
        coordinates = tuple(
            a.ravel() for a in np.meshgrid(prism[0:2], prism[2:4], prism[4:6])
        )
        return coordinates

    def get_easting_edges_center(self, prism):
        """
        Return points on the center of prism edges parallel to easting
        """
        easting_c = (prism[0] + prism[1]) / 2
        northing, upward = tuple(c.ravel() for c in np.meshgrid(prism[2:4], prism[4:6]))
        easting = np.full_like(northing, easting_c)
        return easting, northing, upward

    def get_northing_edges_center(self, prism):
        """
        Return points on the center of prism edges parallel to northing
        """
        northing_c = (prism[2] + prism[3]) / 2
        easting, upward = tuple(c.ravel() for c in np.meshgrid(prism[0:2], prism[4:6]))
        northing = np.full_like(easting, northing_c)
        return easting, northing, upward

    def get_upward_edges_center(self, prism):
        """
        Return points on the center of prism edges parallel to upward
        """
        upward_c = (prism[4] + prism[5]) / 2
        easting, northing = tuple(
            c.ravel() for c in np.meshgrid(prism[0:2], prism[2:4])
        )
        upward = np.full_like(easting, upward_c)
        return easting, northing, upward

    def get_faces_centers(self, prism, direction):
        """
        Return points on the center of faces normal to the given direction
        """
        if direction == "easting":
            easting = prism[0:2]
            northing = np.mean(prism[2:4])
            upward = np.mean(prism[4:6])
        elif direction == "northing":
            easting = np.mean(prism[0:2])
            northing = prism[2:4]
            upward = np.mean(prism[4:6])
        elif direction == "upward":
            easting = np.mean(prism[0:2])
            northing = np.mean(prism[2:4])
            upward = prism[4:6]
        coordinates = tuple(c.ravel() for c in np.meshgrid(easting, northing, upward))
        return coordinates

    def get_interior_points(self, prism):
        """
        Return a set of interior points
        """
        west, east, south, north, bottom, top = prism
        easting = np.linspace(west, east, 5)[1:-1]
        northing = np.linspace(south, north, 5)[1:-1]
        upward = np.linspace(bottom, top, 5)[1:-1]
        coordinates = tuple(c.ravel() for c in np.meshgrid(easting, northing, upward))
        return coordinates

    @pytest.mark.parametrize(
        "forward_func", (magnetic_field, magnetic_e, magnetic_n, magnetic_u)
    )
    def test_on_vertices(self, sample_prism, forward_func):
        """
        Test if magnetic field components on vertices are equal to NaN
        """
        easting, northing, upward = self.get_vertices(sample_prism)
        magnetization = np.array([1.0, 1.0, 1.0])
        results = list(
            forward_func(e, n, u, *sample_prism, *magnetization)
            for (e, n, u) in zip(easting, northing, upward)
        )
        assert np.isnan(results).all()

    @pytest.mark.parametrize(
        "forward_func", (magnetic_field, magnetic_e, magnetic_n, magnetic_u)
    )
    @pytest.mark.parametrize("direction", ("easting", "northing", "upward"))
    def test_on_edges(self, sample_prism, direction, forward_func):
        """
        Test if magnetic field components are NaN on edges of the prism
        """
        # Build observation points on edges
        coordinates = getattr(self, f"get_{direction}_edges_center")(sample_prism)
        easting, northing, upward = coordinates
        magnetization = np.array([1.0, 1.0, 1.0])
        results = list(
            forward_func(e, n, u, *sample_prism, *magnetization)
            for (e, n, u) in zip(easting, northing, upward)
        )
        assert np.isnan(results).all()

    @pytest.mark.parametrize(
        "forward_func", (magnetic_field, magnetic_e, magnetic_n, magnetic_u)
    )
    def test_on_interior_points(self, sample_prism, forward_func):
        """
        Test if magnetic field components are NaN on internal points
        """
        easting, northing, upward = self.get_interior_points(sample_prism)
        magnetization = np.array([1.0, 1.0, 1.0])
        results = list(
            forward_func(e, n, u, *sample_prism, *magnetization)
            for (e, n, u) in zip(easting, northing, upward)
        )
        assert np.isnan(results).all()

    @pytest.mark.parametrize(
        "forward_func", (magnetic_field, magnetic_e, magnetic_n, magnetic_u)
    )
    @pytest.mark.parametrize("direction", ("easting", "northing", "upward"))
    def test_symmetry_on_faces(self, sample_prism, direction, forward_func):
        """
        Tests symmetry of magnetic field components on the center of faces
        normal to the component direction

        For example, check if ``magnetic_u`` has the same value on points in
        the top face and points of the bottom face.
        """
        easting, northing, upward = self.get_faces_centers(sample_prism, direction)
        magnetization = np.array([1.0, 1.0, 1.0])
        if forward_func == magnetic_field:
            component_mapping = {"easting": 0, "northing": 1, "upward": 2}
            index = component_mapping[direction]
            results = list(
                forward_func(e, n, u, *sample_prism, *magnetization)[index]
                for (e, n, u) in zip(easting, northing, upward)
            )
        else:
            results = list(
                forward_func(e, n, u, *sample_prism, *magnetization)
                for (e, n, u) in zip(easting, northing, upward)
            )
        npt.assert_allclose(results, results[0])


class TestMagGradiometryFiniteDifferences:
    """
    Test the magnetic gradiometry components by comparing them to numerical
    derivatives computed through finite differences of the magnetic components.
    """

    delta = 1e-6  # displacement used in the finite difference calculations
    rtol, atol = 5e-4, 5e-12  # tolerances used in the comparisons

    @pytest.mark.parametrize("i", ["e", "n", "u"])
    @pytest.mark.parametrize("j", ["e", "n", "u"])
    def test_derivatives_of_magnetic_components(
        self, sample_3d_grid, sample_prism, sample_magnetization, i, j
    ):
        """
        Compare the magnetic gradiometry functions with a finite difference
        approximation of it computed from the magnetic field components.

        With this function we can compare the result of ``magnetic_{i}{j}``
        against a finite difference approximation using the ``magnetic_{i}``
        forward function, where ``i`` and ``j`` can be ``e``, ``n`` or ``u``.
        """
        # Get forward functions
        mag_func = get_forward_function(i)
        mag_grad_func = get_forward_function(i + j)
        # Evaluate the mag grad function on the sample grid
        mag_grad = evaluate(
            mag_grad_func, sample_3d_grid, sample_prism, sample_magnetization
        )
        # Compute the mag grad function using finite differences
        mag_grad_finite_diff = finite_differences(
            sample_3d_grid,
            sample_prism,
            sample_magnetization,
            direction=j,
            forward_func=mag_func,
            delta=self.delta,
        )
        # Compare results
        npt.assert_allclose(
            mag_grad, mag_grad_finite_diff, rtol=self.rtol, atol=self.atol
        )


class TestMagGradiometryLimits:
    """
    Test if limits are well evaluated in magnetic gradiometry kernels.

    Most of the third-order kernels are singular along the lines that matches
    with the edges of the prism, but the magnetic gradiometry functions are
    defined (and finite) in those regions. These tests ensure that these
    limits are well resolved.
    """

    delta = 1e-6  # displacement used in the finite difference calculations
    rtol, atol = 5e-4, 5e-12  # tolerances used in the comparisons

    def get_coords_on_line(self, prism, direction):
        west, east, south, north, bottom, top = prism
        distance = 13.4  # define distance from the nearest vertex for obs points
        points = []
        if direction == "e":
            for northing, upward in itertools.product((south, north), (bottom, top)):
                points.append([west - distance, northing, upward])
                points.append([east + distance, northing, upward])
        elif direction == "n":
            for easting, upward in itertools.product((west, east), (bottom, top)):
                points.append([easting, south - distance, upward])
                points.append([easting, north + distance, upward])
        elif direction == "u":
            for easting, northing in itertools.product((west, east), (south, north)):
                points.append([easting, northing, bottom - distance])
                points.append([easting, northing, top + distance])
        else:
            raise ValueError(f"Invalid direction '{direction}'.")
        points = np.array(points).T
        easting, northing, upward = points[:]
        return easting, northing, upward

    @pytest.fixture
    def coordinates(self, sample_prism):
        """
        Return a set of coordinates located along the lines of the prism edges.
        """
        west, east, south, north, bottom, top = sample_prism
        distance = 13.4  # define distance from the nearest vertex for obs points
        points = []
        # Define points on lines parallel to upward edges
        for easting, northing in itertools.product((west, east), (south, north)):
            points.append([easting, northing, bottom - distance])
            points.append([easting, northing, top + distance])
        # Define points on lines parallel to easting edges
        for northing, upward in itertools.product((south, north), (bottom, top)):
            points.append([west - distance, northing, upward])
            points.append([east + distance, northing, upward])
        # Define points on lines parallel to northing edges
        for easting, upward in itertools.product((west, east), (bottom, top)):
            points.append([easting, south - distance, upward])
            points.append([easting, north + distance, upward])
        points = np.array(points).T
        easting, northing, upward = points[:]
        return easting, northing, upward

    @pytest.mark.skip
    @pytest.mark.parametrize("i", ["e", "n", "u"])
    @pytest.mark.parametrize("j", ["e", "n", "u"])
    def test_derivatives_of_magnetic_components_on_edges(
        self, coordinates, sample_prism, sample_magnetization, i, j
    ):
        """
        Compare the magnetic gradiometry functions with a finite difference
        approximation of it computed from the magnetic field components.

        With this function we can compare the result of ``magnetic_{i}{j}``
        against a finite difference approximation using the ``magnetic_{i}``
        forward function, where ``i`` and ``j`` can be ``e``, ``n`` or ``u``.
        """
        # Get forward functions
        mag_func = get_forward_function(i)
        mag_grad_func = get_forward_function(i + j)
        # Evaluate the mag grad function on the sample grid
        mag_grad = evaluate(
            mag_grad_func, coordinates, sample_prism, sample_magnetization
        )
        # Compute the mag grad function using finite differences
        mag_grad_finite_diff = finite_differences(
            coordinates,
            sample_prism,
            sample_magnetization,
            direction=j,
            forward_func=mag_func,
            delta=self.delta,
        )
        # Compare results
        npt.assert_allclose(
            mag_grad, mag_grad_finite_diff, rtol=self.rtol, atol=self.atol
        )

    def test_magnetic_ee(self, sample_prism, sample_magnetization):
        """
        Compare the magnetic gradiometry functions with a finite difference
        approximation of it computed from the magnetic field components.

        With this function we can compare the result of ``magnetic_{i}{j}``
        against a finite difference approximation using the ``magnetic_{i}``
        forward function, where ``i`` and ``j`` can be ``e``, ``n`` or ``u``.
        """
        coordinates = self.get_coords_on_line(sample_prism, direction="n")
        # Evaluate the mag grad function on the sample grid
        mag_grad = evaluate(
            magnetic_ee, coordinates, sample_prism, sample_magnetization
        )
        # Compute the mag grad function using finite differences
        mag_grad_finite_diff = finite_differences(
            coordinates,
            sample_prism,
            sample_magnetization,
            direction="e",
            forward_func=magnetic_e,
            delta=self.delta,
        )
        # Compare results
        npt.assert_allclose(
            mag_grad, mag_grad_finite_diff, rtol=self.rtol, atol=self.atol
        )

    def test_magnetic_nn(self, sample_prism, sample_magnetization):
        """
        Compare the magnetic gradiometry functions with a finite difference
        approximation of it computed from the magnetic field components.

        With this function we can compare the result of ``magnetic_{i}{j}``
        against a finite difference approximation using the ``magnetic_{i}``
        forward function, where ``i`` and ``j`` can be ``e``, ``n`` or ``u``.
        """
        coordinates = self.get_coords_on_line(sample_prism, direction="u")
        print(coordinates)
        # Evaluate the mag grad function on the sample grid
        mag_grad = evaluate(
            magnetic_nn, coordinates, sample_prism, sample_magnetization
        )
        # Compute the mag grad function using finite differences
        mag_grad_finite_diff = finite_differences(
            coordinates,
            sample_prism,
            sample_magnetization,
            direction="n",
            forward_func=magnetic_n,
            delta=self.delta,
        )
        # Compare results
        npt.assert_allclose(
            mag_grad, mag_grad_finite_diff, rtol=self.rtol, atol=self.atol
        )
