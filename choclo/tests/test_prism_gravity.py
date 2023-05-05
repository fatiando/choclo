# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test gravity forward modelling functions for rectangular prisms
"""
import numpy as np
import numpy.testing as npt
import pytest

from ..prism import (
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


@pytest.fixture(name="sample_prism_center")
def fixture_sample_prism_center():
    """
    Return the geometric center of the sample prism
    """
    return 30.5, 21.3, -43.5


@pytest.fixture(name="sample_density")
def fixture_sample_density():
    """
    Return the density for the sample prism
    """
    return 400


@pytest.fixture(name="sample_prism_dimensions")
def fixture_sample_prism_dimensions():
    """
    Return the dimensions of the sample prism
    """
    return 10, 15, 20


@pytest.fixture(name="sample_coordinate")
def fixture_sample_coordinate():
    """
    Define a sample observation point
    """
    return (16.7, 13.2, 7.8)


@pytest.fixture(name="sample_prism")
def fixture_sample_prism(sample_prism_center, sample_prism_dimensions):
    """
    Return a sample rectangular prism
    """
    west = sample_prism_center[0] - sample_prism_dimensions[0] / 2
    east = sample_prism_center[0] + sample_prism_dimensions[0] / 2
    south = sample_prism_center[1] - sample_prism_dimensions[1] / 2
    north = sample_prism_center[1] + sample_prism_dimensions[1] / 2
    bottom = sample_prism_center[2] - sample_prism_dimensions[2] / 2
    top = sample_prism_center[2] + sample_prism_dimensions[2] / 2
    prism = [west, east, south, north, bottom, top]
    return np.array(prism)


class TestSymmetryPotential:
    """
    Test the symmetry of gravity_pot of a rectangular prism
    """

    scalers = {
        "inside": 0.8,
        "surface": 1.0,
        "outside": 1.2,
    }

    @pytest.fixture(params=scalers.values(), ids=scalers.keys())
    def coords_in_vertices(self, sample_prism_center, sample_prism_dimensions, request):
        """
        Return observation points located in the vertices of the prism
        """
        # Get the coordinates of the sample prism center
        easting, northing, upward = sample_prism_center
        # Get the dimensions of the sample prism
        d_easting, d_northing, d_upward = sample_prism_dimensions
        # Get scaler
        scaler = request.param
        # Build the vertices
        vertices = list(
            [
                easting + i * scaler * d_easting / 2,
                northing + j * scaler * d_northing / 2,
                upward + k * scaler * d_upward / 2,
            ]
            for i in (-1, 1)
            for j in (-1, 1)
            for k in (-1, 1)
        )
        return vertices

    @pytest.fixture(params=scalers.values(), ids=scalers.keys())
    def coords_in_centers_of_easting_edges(
        self, sample_prism_center, sample_prism_dimensions, request
    ):
        """
        Return observation points located in the center of the prism edges
        parallel to easting axis
        """
        # Get the coordinates of the sample prism center
        easting, northing, upward = sample_prism_center
        # Get the dimensions of the sample prism
        d_easting, _, d_upward = sample_prism_dimensions
        # Get scaler
        scaler = request.param
        # Get the points in the symmetry group
        edges_easting = list(
            [
                easting + i * scaler * d_easting / 2,
                northing,
                upward + k * scaler * d_upward / 2,
            ]
            for i in (-1, 1)
            for k in (-1, 1)
        )
        return edges_easting

    @pytest.fixture(params=scalers.values(), ids=scalers.keys())
    def coords_in_centers_of_northing_edges(
        self, sample_prism_center, sample_prism_dimensions, request
    ):
        """
        Return observation points located in the center of the prism edges
        parallel to northing axis
        """
        # Get the coordinates of the sample prism center
        easting, northing, upward = sample_prism_center
        # Get the dimensions of the sample prism
        _, d_northing, d_upward = sample_prism_dimensions
        # Get scaler
        scaler = request.param
        # Get the points in the symmetry group
        edges_northing = list(
            [
                easting,
                northing + j * scaler * d_northing / 2,
                upward + k * scaler * d_upward / 2,
            ]
            for j in (-1, 1)
            for k in (-1, 1)
        )
        return edges_northing

    @pytest.fixture(params=scalers.values(), ids=scalers.keys())
    def coords_in_centers_of_upward_edges(
        self, sample_prism_center, sample_prism_dimensions, request
    ):
        """
        Return observation points located in the center of the prism edges
        parallel to upward axis
        """
        # Get the coordinates of the sample prism center
        easting, northing, upward = sample_prism_center
        # Get the dimensions of the sample prism
        d_easting, d_northing, _ = sample_prism_dimensions
        # Get scaler
        scaler = request.param
        # Get the points in the symmetry group
        edges_upward = list(
            [
                easting + i * scaler * d_easting / 2,
                northing + j * scaler * d_northing / 2,
                upward,
            ]
            for i in (-1, 1)
            for j in (-1, 1)
        )
        return edges_upward

    @pytest.fixture(params=scalers.values(), ids=scalers.keys())
    def coords_in_centers_of_easting_faces(
        self, sample_prism_center, sample_prism_dimensions, request
    ):
        """
        Return observation points located in the center of the faces normal to
        the easting axis
        """
        # Get the coordinates of the sample prism center
        easting, northing, upward = sample_prism_center
        # Get the dimensions of the sample prism
        d_easting, _, _ = sample_prism_dimensions
        # Get scaler
        scaler = request.param
        # Get the points in the symmetry group
        faces_normal_to_easting = [
            [easting - scaler * d_easting / 2, northing, upward],
            [easting + scaler * d_easting / 2, northing, upward],
        ]
        return faces_normal_to_easting

    @pytest.fixture(params=scalers.values(), ids=scalers.keys())
    def coords_in_centers_of_northing_faces(
        self, sample_prism_center, sample_prism_dimensions, request
    ):
        """
        Return observation points located in the center of the faces normal to
        the northing axis
        """
        # Get the coordinates of the sample prism center
        easting, northing, upward = sample_prism_center
        # Get the dimensions of the sample prism
        _, d_northing, _ = sample_prism_dimensions
        # Get scaler
        scaler = request.param
        # Get the points in the symmetry group
        faces_normal_to_northing = [
            [easting, northing - scaler * d_northing / 2, upward],
            [easting, northing + scaler * d_northing / 2, upward],
        ]
        return faces_normal_to_northing

    @pytest.fixture(params=scalers.values(), ids=scalers.keys())
    def coords_in_centers_of_upward_faces(
        self, sample_prism_center, sample_prism_dimensions, request
    ):
        """
        Return observation points located in the center of the faces normal to
        the northing axis
        """
        # Get the coordinates of the sample prism center
        easting, northing, upward = sample_prism_center
        # Get the dimensions of the sample prism
        _, _, d_upward = sample_prism_dimensions
        # Get scaler
        scaler = request.param
        # Get the points in the symmetry group
        faces_normal_to_upward = [
            [easting, northing, upward - scaler * d_upward / 2],
            [easting, northing, upward + scaler * d_upward / 2],
        ]
        return faces_normal_to_upward

    def test_vertices(self, coords_in_vertices, sample_prism, sample_density):
        """
        Test if gravity_pot satisfies symmetry on vertices
        """
        # Compute gravity_pot on every observation point of the symmetry group
        potential = list(
            gravity_pot(e, n, u, *sample_prism, sample_density)
            for e, n, u in coords_in_vertices
        )
        npt.assert_allclose(potential[0], potential)

    def test_centers_of_easting_edges(
        self, coords_in_centers_of_easting_edges, sample_prism, sample_density
    ):
        """
        Test if gravity_pot satisfies symmetry on centers of the edges
        parallel to the easting direction
        """
        # Compute gravity_pot on every observation point of the symmetry group
        potential = list(
            gravity_pot(e, n, u, *sample_prism, sample_density)
            for e, n, u in coords_in_centers_of_easting_edges
        )
        npt.assert_allclose(potential[0], potential)

    def test_centers_of_northing_edges(
        self, coords_in_centers_of_northing_edges, sample_prism, sample_density
    ):
        """
        Test if gravity_pot satisfies symmetry on centers of the edges
        parallel to the northing direction
        """
        # Compute gravity_pot on every observation point of the symmetry group
        potential = list(
            gravity_pot(e, n, u, *sample_prism, sample_density)
            for e, n, u in coords_in_centers_of_northing_edges
        )
        npt.assert_allclose(potential[0], potential)

    def test_centers_of_upward_edges(
        self, coords_in_centers_of_upward_edges, sample_prism, sample_density
    ):
        """
        Test if gravity_pot satisfies symmetry on centers of the edges
        parallel to the upward direction
        """
        # Compute gravity_pot on every observation point of the symmetry group
        potential = list(
            gravity_pot(e, n, u, *sample_prism, sample_density)
            for e, n, u in coords_in_centers_of_upward_edges
        )
        npt.assert_allclose(potential[0], potential)

    def test_centers_of_easting_faces(
        self, coords_in_centers_of_easting_faces, sample_prism, sample_density
    ):
        """
        Test if gravity_pot satisfies symmetry on centers of the
        centers of the faces normal to the easting direction
        """
        # Compute gravity_pot on every observation point of the symmetry group
        potential = list(
            gravity_pot(e, n, u, *sample_prism, sample_density)
            for e, n, u in coords_in_centers_of_easting_faces
        )
        npt.assert_allclose(potential[0], potential)

    def test_centers_of_northing_faces(
        self, coords_in_centers_of_northing_faces, sample_prism, sample_density
    ):
        """
        Test if gravity_pot satisfies symmetry on centers of the
        centers of the faces normal to the northing direction
        """
        # Compute gravity_pot on every observation point of the symmetry group
        potential = list(
            gravity_pot(e, n, u, *sample_prism, sample_density)
            for e, n, u in coords_in_centers_of_northing_faces
        )
        npt.assert_allclose(potential[0], potential)

    def test_centers_of_upward_faces(
        self, coords_in_centers_of_upward_faces, sample_prism, sample_density
    ):
        """
        Test if gravity_pot satisfies symmetry on centers of the
        centers of the faces normal to the upward direction
        """
        # Compute gravity_pot on every observation point of the symmetry group
        potential = list(
            gravity_pot(e, n, u, *sample_prism, sample_density)
            for e, n, u in coords_in_centers_of_upward_faces
        )
        npt.assert_allclose(potential[0], potential)


class TestSymmetryGravityE:
    """
    Test the symmetry of gravity_e of a rectangular prism
    """

    @pytest.fixture()
    def coords_in_northing_upward_plane(
        self, sample_prism_center, sample_prism_dimensions
    ):
        """
        Return observation points located in the northing-upward plane that
        pass along the center of the prism.
        """
        # Get the coordinates of the sample prism center
        center_easting, center_northing, center_upward = sample_prism_center
        # Get the dimensions of the sample prism
        _, d_northing, d_upward = sample_prism_dimensions
        # Build the points
        n_per_side = 5
        max_northing = d_northing / 2 * n_per_side
        max_upward = d_upward / 2 * n_per_side
        northing = np.linspace(-max_northing, max_northing, 2 * n_per_side + 1)
        upward = np.linspace(-max_upward, max_upward, 2 * n_per_side + 1)
        # Shift coordinates
        easting = np.zeros_like(northing) + center_easting
        northing += center_northing
        upward += center_upward
        return easting, northing, upward

    @pytest.fixture
    def mirrored_points(self, sample_prism_center, sample_prism_dimensions):
        """
        Define two set of mirrored points across the northing-upward plane
        that passes through the prism center
        """
        # Get the coordinates of the sample prism center
        center_easting, center_northing, center_upward = sample_prism_center
        # Get the dimensions of the sample prism
        d_easting, d_northing, d_upward = sample_prism_dimensions
        # Build the points
        n_per_side = 4
        max_easting = d_easting / 2 * n_per_side
        max_northing = d_northing / 2 * n_per_side
        max_upward = d_upward / 2 * n_per_side
        easting_east = np.linspace(d_easting / 2, max_easting, n_per_side)
        northing = np.linspace(-max_northing, max_northing, 2 * n_per_side + 1)
        upward = np.linspace(-max_upward, max_upward, 2 * n_per_side + 1)
        # Meshgrid
        easting_east, northing, upward = (
            array.ravel() for array in np.meshgrid(easting_east, northing, upward)
        )
        # Define easting_west
        easting_west = -easting_east
        # Shift coordinates
        easting_west += center_easting
        easting_east += center_easting
        northing += center_northing
        upward += center_upward
        return (easting_west, northing, upward), (easting_east, northing, upward)

    def test_northing_upward_plane(
        self,
        coords_in_northing_upward_plane,
        sample_prism,
        sample_prism_center,
        sample_density,
    ):
        """
        Test if gravity_e is null in points of the northing-upward plane that
        passes through its center.
        """
        # Compute gravity_e on every observation point of northing-upward plane
        g_e = list(
            gravity_e(e, n, u, *sample_prism, sample_density)
            for e, n, u in zip(*coords_in_northing_upward_plane)
        )
        # Compute gravity_e on a point slightly shifted from the prism center
        # (it will be our control for non-zero field)
        easting, northing, upward = sample_prism_center
        non_zero_g_e = gravity_e(
            easting + 1e-10, northing, upward, *sample_prism, sample_density
        )
        atol = np.abs(non_zero_g_e).max()
        npt.assert_allclose(g_e, 0, atol=atol)

    def test_mirrored_points(self, mirrored_points, sample_prism, sample_density):
        """
        Test if gravity_e is opposite in mirrored points across the
        northing-upward plane
        """
        west, east = mirrored_points
        # Compute gravity_e on every observation point of the two planes
        g_e_west = np.array(
            list(
                gravity_e(e, n, u, *sample_prism, sample_density)
                for e, n, u in zip(*west)
            )
        )
        g_e_east = np.array(
            list(
                gravity_e(e, n, u, *sample_prism, sample_density)
                for e, n, u in zip(*east)
            )
        )
        npt.assert_allclose(g_e_west, -g_e_east)

    @pytest.mark.parametrize(
        "density", (-200, 200), ids=("negative_density", "positive_density")
    )
    def test_sign(self, sample_prism_center, sample_prism, density):
        """
        Test that gravity_e has the correct sign on west and east points
        """
        # Get easting dimension of the prism
        d_easting = sample_prism[1] - sample_prism[0]
        # Define an observation point on the east of the prism
        point = (
            sample_prism_center[0] + d_easting,
            sample_prism_center[1],
            sample_prism_center[2],
        )
        # Compute gravity_e
        g_e = gravity_e(*point, *sample_prism, density)
        # Check if g_e has the opposite sign as its density (the acceleration
        # will point westward for a positive density, therefore negative
        # easting component)
        assert np.sign(g_e) != np.sign(density)

    def test_on_vertices(self, sample_prism, sample_density):
        """
        Test g_e on the vertices
        """
        # Get boundaries of the prism
        west, east, south, north, bottom, top = sample_prism
        # Split the vertices of the prism between the ones in the west and
        # east faces
        east_vertices = [[east, n, u] for n in (south, north) for u in (bottom, top)]
        west_vertices = [[west, n, u] for n in (south, north) for u in (bottom, top)]
        g_e_east = np.array(
            [
                gravity_e(*point, *sample_prism, sample_density)
                for point in east_vertices
            ]
        )
        g_e_west = np.array(
            [
                gravity_e(*point, *sample_prism, sample_density)
                for point in west_vertices
            ]
        )
        npt.assert_allclose(g_e_east, -g_e_west)


class TestSymmetryGravityN:
    """
    Test the symmetry of gravity_n of a rectangular prism
    """

    @pytest.fixture()
    def coords_in_easting_upward_plane(
        self, sample_prism_center, sample_prism_dimensions
    ):
        """
        Return observation points located in the easting-upward plane that
        pass along the center of the prism.
        """
        # Get the coordinates of the sample prism center
        center_easting, center_northing, center_upward = sample_prism_center
        # Get the dimensions of the sample prism
        d_easting, _, d_upward = sample_prism_dimensions
        # Build the points
        n_per_side = 5
        max_easting = d_easting / 2 * n_per_side
        max_upward = d_upward / 2 * n_per_side
        easting = np.linspace(-max_easting, max_easting, 2 * n_per_side + 1)
        upward = np.linspace(-max_upward, max_upward, 2 * n_per_side + 1)
        # Shift coordinates
        easting += center_easting
        northing = np.zeros_like(easting) + center_northing
        upward += center_upward
        return easting, northing, upward

    @pytest.fixture
    def mirrored_points(self, sample_prism_center, sample_prism_dimensions):
        """
        Define two set of mirrored points across the easting-upward plane
        that passes through the prism center
        """
        # Get the coordinates of the sample prism center
        center_easting, center_northing, center_upward = sample_prism_center
        # Get the dimensions of the sample prism
        d_easting, d_northing, d_upward = sample_prism_dimensions
        # Build the points
        n_per_side = 4
        max_easting = d_easting / 2 * n_per_side
        max_northing = d_northing / 2 * n_per_side
        max_upward = d_upward / 2 * n_per_side
        easting = np.linspace(-max_easting, max_easting, 2 * n_per_side + 1)
        northing_north = np.linspace(d_northing / 2, max_northing, n_per_side)
        upward = np.linspace(-max_upward, max_upward, 2 * n_per_side + 1)
        # Meshgrid
        easting, northing_north, upward = (
            array.ravel() for array in np.meshgrid(easting, northing_north, upward)
        )
        # Define northing_south
        northing_south = -northing_north
        # Shift coordinates
        easting += center_easting
        northing_north += center_northing
        northing_south += center_northing
        upward += center_upward
        return (easting, northing_south, upward), (easting, northing_north, upward)

    def test_northing_upward_plane(
        self,
        coords_in_easting_upward_plane,
        sample_prism,
        sample_prism_center,
        sample_density,
    ):
        """
        Test if gravity_n is null in points of the easting-upward plane that
        passes through its center.
        """
        # Compute gravity_n on every observation point of easting-upward plane
        g_n = list(
            gravity_n(e, n, u, *sample_prism, sample_density)
            for e, n, u in zip(*coords_in_easting_upward_plane)
        )
        # Compute gravity_n on a point slightly shifted from the prism center
        # (it will be our control for non-zero field)
        easting, northing, upward = sample_prism_center
        non_zero_g_n = gravity_n(
            easting, northing + 1e-10, upward, *sample_prism, sample_density
        )
        atol = np.abs(non_zero_g_n).max()
        npt.assert_allclose(g_n, 0, atol=atol)

    def test_mirrored_points(self, mirrored_points, sample_prism, sample_density):
        """
        Test if gravity_n is opposite in mirrored points across the
        easting-upward plane
        """
        south, north = mirrored_points
        # Compute gravity_n on every observation point of the two planes
        g_n_south = np.array(
            list(
                gravity_n(e, n, u, *sample_prism, sample_density)
                for e, n, u in zip(*south)
            )
        )
        g_n_north = np.array(
            list(
                gravity_n(e, n, u, *sample_prism, sample_density)
                for e, n, u in zip(*north)
            )
        )
        npt.assert_allclose(g_n_south, -g_n_north)

    @pytest.mark.parametrize(
        "density", (-200, 200), ids=("negative_density", "positive_density")
    )
    def test_sign(self, sample_prism_center, sample_prism, density):
        """
        Test that gravity_n has the correct sign on south and north points
        """
        # Get easting dimension of the prism
        d_northing = sample_prism[3] - sample_prism[2]
        # Define an observation point on the north of the prism
        point = (
            sample_prism_center[0],
            sample_prism_center[1] + d_northing,
            sample_prism_center[2],
        )
        # Compute gravity_e
        g_n = gravity_n(*point, *sample_prism, density)
        # Check if g_n has the opposite sign as its density (the acceleration
        # will point southward for a positive density, therefore negative
        # northing component)
        assert np.sign(g_n) != np.sign(density)

    def test_on_vertices(self, sample_prism, sample_density):
        """
        Test g_n on the vertices
        """
        # Get boundaries of the prism
        west, east, south, north, bottom, top = sample_prism
        # Split the vertices of the prism between the ones in the south and
        # north faces
        north_vertices = [[e, south, u] for e in (west, east) for u in (bottom, top)]
        south_vertices = [[e, north, u] for e in (west, east) for u in (bottom, top)]
        g_n_north = np.array(
            [
                gravity_n(*point, *sample_prism, sample_density)
                for point in north_vertices
            ]
        )
        g_n_south = np.array(
            [
                gravity_n(*point, *sample_prism, sample_density)
                for point in south_vertices
            ]
        )
        npt.assert_allclose(g_n_north, -g_n_south)


class TestSymmetryGravityU:
    """
    Test the symmetry of gravity_u of a rectangular prism
    """

    @pytest.fixture()
    def coords_in_easting_northing_plane(
        self, sample_prism_center, sample_prism_dimensions
    ):
        """
        Return observation points located in the easting-northing plane that
        pass along the center of the prism.
        """
        # Get the coordinates of the sample prism center
        center_easting, center_northing, center_upward = sample_prism_center
        # Get the dimensions of the sample prism
        d_easting, d_northing, _ = sample_prism_dimensions
        # Build the points
        n_per_side = 5
        max_easting = d_easting / 2 * n_per_side
        max_northing = d_northing / 2 * n_per_side
        easting = np.linspace(-max_easting, max_easting, 2 * n_per_side + 1)
        northing = np.linspace(-max_northing, max_northing, 2 * n_per_side + 1)
        # Shift coordinates
        easting += center_easting
        northing += center_northing
        upward = np.zeros_like(easting) + center_upward
        return easting, northing, upward

    @pytest.fixture
    def mirrored_points(self, sample_prism_center, sample_prism_dimensions):
        """
        Define two set of mirrored points across the easting-northing plane
        that passes through the prism center
        """
        # Get the coordinates of the sample prism center
        center_easting, center_northing, center_upward = sample_prism_center
        # Get the dimensions of the sample prism
        d_easting, d_northing, d_upward = sample_prism_dimensions
        # Build the points
        n_per_side = 4
        max_easting = d_easting / 2 * n_per_side
        max_northing = d_northing / 2 * n_per_side
        max_upward = d_upward / 2 * n_per_side
        easting = np.linspace(-max_easting, max_easting, 2 * n_per_side + 1)
        northing = np.linspace(-max_northing, max_northing, 2 * n_per_side + 1)
        upward_top = np.linspace(d_upward / 2, max_upward, n_per_side)
        # Meshgrid
        easting, northing, upward_top = (
            array.ravel() for array in np.meshgrid(easting, northing, upward_top)
        )
        # Define upward bottom
        upward_bottom = -upward_top
        # Shift coordinates
        easting += center_easting
        northing += center_northing
        upward_top += center_upward
        upward_bottom += center_upward
        return (easting, northing, upward_top), (easting, northing, upward_bottom)

    def test_easting_northing_plane(
        self,
        coords_in_easting_northing_plane,
        sample_prism,
        sample_prism_center,
        sample_density,
    ):
        """
        Test if gravity_u is null in points of the easting-northing plane that
        passes through its center.
        """
        # Compute gravity_u on every observation point of the easting-northing
        # plane
        g_u = list(
            gravity_u(e, n, u, *sample_prism, sample_density)
            for e, n, u in zip(*coords_in_easting_northing_plane)
        )
        # Compute gravity_u on a point slightly shifted from the prism center
        # (it will be our control for non-zero field)
        easting, northing, upward = sample_prism_center
        non_zero_g_u = gravity_u(
            easting, northing, upward + 1e-10, *sample_prism, sample_density
        )
        atol = np.abs(non_zero_g_u).max()
        npt.assert_allclose(g_u, 0, atol=atol)

    def test_mirrored_points(self, mirrored_points, sample_prism, sample_density):
        """
        Test if gravity_u is opposite in mirrored points across the
        easting-northing plane
        """
        top, bottom = mirrored_points
        # Compute gravity_u on every observation point of the two planes
        g_u_bottom = np.array(
            list(
                gravity_u(e, n, u, *sample_prism, sample_density)
                for e, n, u in zip(*bottom)
            )
        )
        g_u_top = np.array(
            list(
                gravity_u(e, n, u, *sample_prism, sample_density)
                for e, n, u in zip(*top)
            )
        )
        npt.assert_allclose(g_u_bottom, -g_u_top)

    @pytest.mark.parametrize(
        "density", (-200, 200), ids=("negative_density", "positive_density")
    )
    def test_sign(self, sample_prism_center, sample_prism, density):
        """
        Test that gravity_u has the correct sign on top and bottom points
        """
        # Get upward dimension of the prism
        d_upward = sample_prism[5] - sample_prism[4]
        # Define an observation point on top of the prism
        point = (
            sample_prism_center[0],
            sample_prism_center[1],
            sample_prism_center[2] + d_upward,
        )
        # Compute gravity_u
        g_u = gravity_u(*point, *sample_prism, density)
        # Check if g_u has the opposite sign as its density (the acceleration
        # will point downwards for a positive density, therefore negative
        # upward component)
        assert np.sign(g_u) != np.sign(density)

    def test_on_vertices(self, sample_prism, sample_density):
        """
        Test g_u on the vertices
        """
        # Get boundaries of the prism
        west, east, south, north, bottom, top = sample_prism
        # Split the vertices of the prism between the ones in the top and
        # bottom faces
        top_vertices = [[e, n, top] for e in (west, east) for n in (south, north)]
        bottom_vertices = [[e, n, bottom] for e in (west, east) for n in (south, north)]
        g_u_top = np.array(
            [gravity_u(*point, *sample_prism, sample_density) for point in top_vertices]
        )
        g_u_bottom = np.array(
            [
                gravity_u(*point, *sample_prism, sample_density)
                for point in bottom_vertices
            ]
        )
        npt.assert_allclose(g_u_top, -g_u_bottom)


class TestAccelerationFiniteDifferences:
    """
    Test acceleration components against finite-differences approximations of
    the gravity potential
    """

    # Define a small displacement for computing the finite differences
    delta = 0.0001  # 0.1 mm

    # Define expected relative error tolerance in the comparisons
    rtol = 1e-5

    @pytest.fixture
    def finite_diff_gravity_e(self, sample_coordinate, sample_prism, sample_density):
        """
        Compute gravity_e through finite differences of the gravity_pot
        """
        easting_p, northing_p, upward_p = sample_coordinate
        west, east = sample_prism[:2]
        # Compute shifted coordinate
        shifted_coordinate = (easting_p + self.delta, northing_p, upward_p)
        # Calculate g_e through finite differences
        g_e = (
            gravity_pot(*shifted_coordinate, *sample_prism, sample_density)
            - gravity_pot(*sample_coordinate, *sample_prism, sample_density)
        ) / self.delta
        return g_e

    @pytest.fixture
    def finite_diff_gravity_n(self, sample_coordinate, sample_prism, sample_density):
        """
        Compute gravity_n through finite differences of the gravity_pot
        """
        easting_p, northing_p, upward_p = sample_coordinate
        south, north = sample_prism[2:4]
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p + self.delta, upward_p)
        # Calculate g_n through finite differences
        g_n = (
            gravity_pot(*shifted_coordinate, *sample_prism, sample_density)
            - gravity_pot(*sample_coordinate, *sample_prism, sample_density)
        ) / self.delta
        return g_n

    @pytest.fixture
    def finite_diff_gravity_u(self, sample_coordinate, sample_prism, sample_density):
        """
        Compute gravity_u through finite differences of the gravity_pot
        """
        easting_p, northing_p, upward_p = sample_coordinate
        bottom, top = sample_prism[-2:]
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p, upward_p + self.delta)
        # Calculate g_u through finite differences
        g_u = (
            gravity_pot(*shifted_coordinate, *sample_prism, sample_density)
            - gravity_pot(*sample_coordinate, *sample_prism, sample_density)
        ) / self.delta
        return g_u

    def test_gravity_e(
        self, sample_coordinate, sample_prism, sample_density, finite_diff_gravity_e
    ):
        """
        Test gravity_e against finite differences of the gravity_pot
        """
        g_e = gravity_e(*sample_coordinate, *sample_prism, sample_density)
        npt.assert_allclose(g_e, finite_diff_gravity_e, rtol=self.rtol)

    def test_gravity_n(
        self, sample_coordinate, sample_prism, sample_density, finite_diff_gravity_n
    ):
        """
        Test gravity_n against finite differences of the gravity_pot
        """
        g_n = gravity_n(*sample_coordinate, *sample_prism, sample_density)
        npt.assert_allclose(g_n, finite_diff_gravity_n, rtol=self.rtol)

    def test_gravity_u(
        self, sample_coordinate, sample_prism, sample_density, finite_diff_gravity_u
    ):
        """
        Test gravity_u against finite differences of the gravity_pot
        """
        g_u = gravity_u(*sample_coordinate, *sample_prism, sample_density)
        npt.assert_allclose(g_u, finite_diff_gravity_u, rtol=self.rtol)


class TestTensorFiniteDifferences:
    """
    Test tensor components against finite-differences approximations of
    the gravity gradient
    """

    # Define a small displacement for computing the finite differences
    delta = 0.0001  # 0.1 mm

    # Define expected relative error tolerance in the comparisons
    rtol = 1e-5

    @pytest.fixture
    def finite_diff_gravity_ee(self, sample_coordinate, sample_prism, sample_density):
        """
        Compute gravity_ee through finite differences of the gravity_e
        """
        easting_p, northing_p, upward_p = sample_coordinate
        west, east = sample_prism[:2]
        # Compute shifted coordinate
        shifted_coordinate = (easting_p + self.delta, northing_p, upward_p)
        # Calculate g_ee through finite differences
        g_ee = (
            gravity_e(*shifted_coordinate, *sample_prism, sample_density)
            - gravity_e(*sample_coordinate, *sample_prism, sample_density)
        ) / self.delta
        return g_ee

    @pytest.fixture
    def finite_diff_gravity_nn(self, sample_coordinate, sample_prism, sample_density):
        """
        Compute gravity_nn through finite differences of the gravity_n
        """
        easting_p, northing_p, upward_p = sample_coordinate
        south, north = sample_prism[2:4]
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p + self.delta, upward_p)
        # Calculate g_nn through finite differences
        g_nn = (
            gravity_n(*shifted_coordinate, *sample_prism, sample_density)
            - gravity_n(*sample_coordinate, *sample_prism, sample_density)
        ) / self.delta
        return g_nn

    @pytest.fixture
    def finite_diff_gravity_uu(self, sample_coordinate, sample_prism, sample_density):
        """
        Compute gravity_uu through finite differences of the gravity_u
        """
        easting_p, northing_p, upward_p = sample_coordinate
        bottom, top = sample_prism[-2:]
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p, upward_p + self.delta)
        # Calculate g_u through finite differences
        g_uu = (
            gravity_u(*shifted_coordinate, *sample_prism, sample_density)
            - gravity_u(*sample_coordinate, *sample_prism, sample_density)
        ) / self.delta
        return g_uu

    @pytest.fixture
    def finite_diff_gravity_en(self, sample_coordinate, sample_prism, sample_density):
        """
        Compute gravity_en through finite differences of the gravity_e
        """
        easting_p, northing_p, upward_p = sample_coordinate
        west, east = sample_prism[:2]
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p + self.delta, upward_p)
        # Calculate g_en through finite differences
        g_en = (
            gravity_e(*shifted_coordinate, *sample_prism, sample_density)
            - gravity_e(*sample_coordinate, *sample_prism, sample_density)
        ) / self.delta
        return g_en

    @pytest.fixture
    def finite_diff_gravity_eu(self, sample_coordinate, sample_prism, sample_density):
        """
        Compute gravity_eu through finite differences of the gravity_e
        """
        easting_p, northing_p, upward_p = sample_coordinate
        west, east = sample_prism[:2]
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p, upward_p + self.delta)
        # Calculate g_en through finite differences
        g_eu = (
            gravity_e(*shifted_coordinate, *sample_prism, sample_density)
            - gravity_e(*sample_coordinate, *sample_prism, sample_density)
        ) / self.delta
        return g_eu

    @pytest.fixture
    def finite_diff_gravity_nu(self, sample_coordinate, sample_prism, sample_density):
        """
        Compute gravity_nu through finite differences of the gravity_n
        """
        easting_p, northing_p, upward_p = sample_coordinate
        west, east = sample_prism[:2]
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p, upward_p + self.delta)
        # Calculate g_en through finite differences
        g_nu = (
            gravity_n(*shifted_coordinate, *sample_prism, sample_density)
            - gravity_n(*sample_coordinate, *sample_prism, sample_density)
        ) / self.delta
        return g_nu

    def test_gravity_ee(
        self, sample_coordinate, sample_prism, sample_density, finite_diff_gravity_ee
    ):
        """
        Test gravity_ee against finite differences of the gravity_e
        """
        g_ee = gravity_ee(*sample_coordinate, *sample_prism, sample_density)
        npt.assert_allclose(g_ee, finite_diff_gravity_ee, rtol=self.rtol)

    def test_gravity_nn(
        self, sample_coordinate, sample_prism, sample_density, finite_diff_gravity_nn
    ):
        """
        Test gravity_nn against finite differences of the gravity_n
        """
        g_nn = gravity_nn(*sample_coordinate, *sample_prism, sample_density)
        npt.assert_allclose(g_nn, finite_diff_gravity_nn, rtol=self.rtol)

    def test_gravity_uu(
        self, sample_coordinate, sample_prism, sample_density, finite_diff_gravity_uu
    ):
        """
        Test gravity_uu against finite differences of the gravity_u
        """
        g_uu = gravity_uu(*sample_coordinate, *sample_prism, sample_density)
        npt.assert_allclose(g_uu, finite_diff_gravity_uu, rtol=self.rtol)

    def test_gravity_en(
        self, sample_coordinate, sample_prism, sample_density, finite_diff_gravity_en
    ):
        """
        Test gravity_en against finite differences of the gravity_e
        """
        g_en = gravity_en(*sample_coordinate, *sample_prism, sample_density)
        npt.assert_allclose(g_en, finite_diff_gravity_en, rtol=self.rtol)

    def test_gravity_eu(
        self, sample_coordinate, sample_prism, sample_density, finite_diff_gravity_eu
    ):
        """
        Test gravity_eu against finite differences of the gravity_e
        """
        g_eu = gravity_eu(*sample_coordinate, *sample_prism, sample_density)
        npt.assert_allclose(g_eu, finite_diff_gravity_eu, rtol=self.rtol)

    def test_gravity_nu(
        self, sample_coordinate, sample_prism, sample_density, finite_diff_gravity_nu
    ):
        """
        Test gravity_nu against finite differences of the gravity_n
        """
        g_nu = gravity_nu(*sample_coordinate, *sample_prism, sample_density)
        npt.assert_allclose(g_nu, finite_diff_gravity_nu, rtol=self.rtol)


class TestLaplacian:
    @pytest.fixture
    def sample_observation_points(self, sample_prism_center, sample_prism_dimensions):
        """
        Return a 3D grid of observation points around the sample prism.
        The grid doesn't have points on edges and vertices (where the tensor
        components are not defined) and doesn't include the prism center where
        the fields will be zero.
        """
        # Get the coordinates of the sample prism center
        center_easting, center_northing, center_upward = sample_prism_center
        # Get the dimensions of the sample prism
        d_easting, d_northing, d_upward = sample_prism_dimensions
        # Build the points (avoid points in edges or vertices where the tensor
        # components are not defined)
        n_per_side = 4
        max_easting = 2 / 3 * d_easting * n_per_side
        max_northing = 2 / 3 * d_northing * n_per_side
        max_upward = 2 / 3 * d_upward * n_per_side
        easting = np.linspace(-max_easting, max_easting, 2 * n_per_side + 1)
        northing = np.linspace(-max_northing, max_northing, 2 * n_per_side + 1)
        upward = np.linspace(-max_upward, max_upward, 2 * n_per_side + 1)
        # Get the meshgrid
        easting, northing, upward = tuple(
            a.ravel() for a in np.meshgrid(easting, northing, upward)
        )
        # Remove the center of the prism
        is_prism_center = (easting == 0) & (northing == 0) & (upward == 0)
        easting, northing, upward = tuple(
            a[np.logical_not(is_prism_center)] for a in (easting, northing, upward)
        )
        # Shift the coordinates
        easting += center_easting
        northing += center_northing
        upward += center_upward
        return easting, northing, upward

    @pytest.mark.parametrize("left_component", ("g_ee", "g_nn", "g_uu"))
    def test_laplacian(
        self, sample_observation_points, sample_prism, sample_density, left_component
    ):
        """
        Test if diagonal tensor functions satisfy Laplace's equation
        """
        g_ee = np.array(
            [
                gravity_ee(e, n, u, *sample_prism, sample_density)
                for e, n, u in zip(*sample_observation_points)
            ]
        )
        g_nn = np.array(
            [
                gravity_nn(e, n, u, *sample_prism, sample_density)
                for e, n, u in zip(*sample_observation_points)
            ]
        )
        g_uu = np.array(
            [
                gravity_uu(e, n, u, *sample_prism, sample_density)
                for e, n, u in zip(*sample_observation_points)
            ]
        )
        if left_component == "g_ee":
            npt.assert_allclose(-g_ee, g_nn + g_uu)
        if left_component == "g_nn":
            npt.assert_allclose(-g_nn, g_ee + g_uu)
        if left_component == "g_uu":
            npt.assert_allclose(-g_uu, g_ee + g_nn)


class TestNonDiagonalTensor:
    """
    Test the behaviour of non-diagonal tensor components on critical points

    The non-diagonal tensor components evaluate the log function on
    ``log(x + r)``, where ``x`` is the shifted coordinate of the vertex and
    ``r`` is the distance between the vertex and the computation point. When
    the two other shifted coordinates are zero and ``x`` is negative, then
    ``x == -r``, making impossible to evaluate the log function.
    The ``_safe_log`` function accounts for this and safely evaluate the log
    function in these points.

    If we consider for example the ``g_en`` component, then observation points
    that fall above one of the vertices of the prism lead to this situation. If
    the ``_safe_log`` function is not properly defined, then the value of
    ``g_en`` on those points will be significantly different from the
    surrounding points, even with a different sign.

    The following test functions compare the values of the non-diagonal
    components on and around these particular observation points.
    """

    @pytest.fixture(name="prism")
    def prism(self):
        prism = [-10, 10, -20, 20, -50, -15]
        return np.array(prism, dtype=np.float64)

    @pytest.fixture(name="density")
    def density(self):
        return 400.0

    @pytest.mark.parametrize("easting_boundary", (0, 1))
    @pytest.mark.parametrize("northing_boundary", (2, 3))
    def test_g_en_above_vertices(
        self, prism, density, easting_boundary, northing_boundary
    ):
        """
        Test g_en on an observation point above the node and around it
        """
        vertex_easting = prism[easting_boundary]
        vertex_northing = prism[northing_boundary]
        # Consider an observation point above one of the vertex of the prism
        # and define a few observation points around it
        easting = np.linspace(vertex_easting - 3, vertex_easting + 3, 61)
        northing = np.linspace(vertex_northing - 3, vertex_northing + 3, 61)
        assert vertex_easting in easting and vertex_northing in northing
        upward = prism[5] + 1  # locate the observation points above the prism
        g_en = np.array(
            [
                gravity_en(e, n, upward, *prism, density)
                for e in easting
                for n in northing
            ]
        )
        # Check if all values in g_en have the same sign
        signs = np.sign(g_en)
        npt.assert_allclose(signs[0], signs)

    @pytest.mark.parametrize("easting_boundary", (0, 1))
    @pytest.mark.parametrize("upward_boundary", (4, 5))
    def test_g_eu_north_vertices(
        self, prism, density, easting_boundary, upward_boundary
    ):
        """
        Test g_eu on an observation point north the node and around it
        """
        vertex_easting = prism[easting_boundary]
        vertex_upward = prism[upward_boundary]
        # Consider an observation point at the north of one of the vertex of
        # the prism and define a few observation points around it
        easting = np.linspace(vertex_easting - 3, vertex_easting + 3, 61)
        upward = np.linspace(vertex_upward - 3, vertex_upward + 3, 61)
        assert vertex_easting in easting and vertex_upward in upward
        northing = prism[3] + 1  # locate the observation points north the prism
        g_eu = np.array(
            [
                gravity_eu(e, northing, u, *prism, density)
                for e in easting
                for u in upward
            ]
        )
        # Check if all values in g_eu have the same sign
        signs = np.sign(g_eu)
        npt.assert_allclose(signs[0], signs)

    @pytest.mark.parametrize("northing_boundary", (2, 3))
    @pytest.mark.parametrize("upward_boundary", (4, 5))
    def test_g_nu_east_vertices(
        self, prism, density, northing_boundary, upward_boundary
    ):
        """
        Test g_nu on an observation point north the node and around it
        """
        vertex_northing = prism[northing_boundary]
        vertex_upward = prism[upward_boundary]
        # Consider an observation point at the east of one of the vertex of
        # the prism and define a few observation points around it
        northing = np.linspace(vertex_northing - 3, vertex_northing + 3, 61)
        upward = np.linspace(vertex_upward - 3, vertex_upward + 3, 61)
        assert vertex_northing in northing and vertex_upward in upward
        easting = prism[1] + 1  # locate the observation points north the prism
        g_nu = np.array(
            [
                gravity_nu(easting, n, u, *prism, density)
                for n in northing
                for u in upward
            ]
        )
        # Check if all values in g_nu have the same sign
        signs = np.sign(g_nu)
        npt.assert_allclose(signs[0], signs)


class TestNonDiagonalTensorSymmetry:
    """
    Test symmetry of non-diagonal tensor components

    Each test will compute one tensor component on two regular grids. Each grid
    falls in one of two parallel planes that are equidistant to the prism. The
    grids will contain observation points that share a pair of coordinates
    with the vertices of the prism to test symmetry on those regions.
    For example, for `g_en` two horizontal grids will be defined: one above and
    one below the prism. The first one will contain points that fall right
    above its vertices, while the second one will contain points that fall
    right below them (these points share the easting and northing coordinates
    with these vertices).

    For the non-diagonal tensor components these points are where the
    modifications of the log functions take protagonism.
    """

    atol = 1e-18

    @pytest.fixture(name="prism")
    def prism(self):
        prism = [-10, 10, -20, 20, -30, 30]
        return np.array(prism, dtype=np.float64)

    @pytest.fixture(name="density")
    def density(self):
        return 400.0

    def test_g_en_symmetry(self, prism, density):
        """
        Test symmetry of g_en
        """
        west, east, south, north, bottom, top = prism[:]
        # Define two horizontal grids
        # (make sure that contain points that fall above and below the
        # vertices)
        easting = np.linspace(-40, 40, 41)
        northing = np.linspace(-40, 40, 41)
        assert west in easting and east in easting
        assert south in northing and north in northing
        delta = 2
        g_en_top = np.array(
            [
                gravity_en(e, n, top + delta, *prism, density)
                for e in easting
                for n in northing
            ]
        )
        g_en_bottom = np.array(
            [
                gravity_en(e, n, bottom - delta, *prism, density)
                for e in easting
                for n in northing
            ]
        )
        npt.assert_allclose(g_en_top, g_en_bottom, atol=self.atol)

    def test_g_eu_symmetry(self, prism, density):
        """
        Test symmetry of g_eu
        """
        west, east, south, north, bottom, top = prism[:]
        # Define two vertical grids parallel to the easting-upward plane
        # (make sure that contain points that fall north and south the
        # vertices)
        easting = np.linspace(-40, 40, 41)
        upward = np.linspace(-40, 40, 41)
        assert west in easting and east in easting
        assert bottom in upward and top in upward
        delta = 2
        g_eu_north = np.array(
            [
                gravity_eu(e, north + delta, u, *prism, density)
                for e in easting
                for u in upward
            ]
        )
        g_eu_south = np.array(
            [
                gravity_eu(e, south - delta, u, *prism, density)
                for e in easting
                for u in upward
            ]
        )
        npt.assert_allclose(g_eu_north, g_eu_south, atol=self.atol)

    def test_g_nu_symmetry(self, prism, density):
        """
        Test symmetry of g_nu
        """
        west, east, south, north, bottom, top = prism[:]
        # Define two vertical grids parallel to the northing-upward plane
        # (make sure that contain points that fall north and south the
        # vertices)
        northing = np.linspace(-40, 40, 41)
        upward = np.linspace(-40, 40, 41)
        assert south in northing and north in northing
        assert bottom in upward and top in upward
        delta = 2
        g_nu_north = np.array(
            [
                gravity_nu(east + delta, n, u, *prism, density)
                for n in northing
                for u in upward
            ]
        )
        g_nu_south = np.array(
            [
                gravity_nu(west - delta, n, u, *prism, density)
                for n in northing
                for u in upward
            ]
        )
        npt.assert_allclose(g_nu_north, g_nu_south, atol=self.atol)


class TestDiagonalTensorSingularities:
    """
    Test if diagonal tensor components behave as expected on their singular
    points

    Diagonal tensor components have singular points on:
    * prism vertices,
    * prism edges perpendicular to the tensor direction, and
    * prism faces normal to the tensor direction.

    For the first two cases, the forward modelling function should return
    ``np.nan``. For the last case, it should return the limit of the field when
    we approach from outside of the prism.
    """

    @pytest.fixture()
    def prism_boundaries(self):
        """
        Return the boundaries of the sample prism
        """
        west, east, south, north, bottom, top = -5.4, 10.1, 43.2, 79.5, -53.7, -44.3
        return west, east, south, north, bottom, top

    @pytest.mark.parametrize("function", (gravity_ee, gravity_nn, gravity_uu))
    def test_on_vertices(self, prism_boundaries, function):
        """
        Test if diagonal tensor components on vertices are equal to NaN
        """
        west, east, south, north, bottom, top = prism_boundaries
        prism = np.array([west, east, south, north, bottom, top])
        density = 1.0
        coordinates = tuple(
            a.ravel() for a in np.meshgrid([west, east], [south, north], [bottom, top])
        )
        results = list(
            function(e, n, u, *prism, density) for (e, n, u) in zip(*coordinates)
        )
        assert np.isnan(results).all()

    def test_gee_on_edges(self, prism_boundaries):
        """
        Test if gravity_ee on edges perpendicular to easting is equal to NaN
        """
        # Define prism
        west, east, south, north, bottom, top = prism_boundaries
        prism = np.array([west, east, south, north, bottom, top])
        density = 1.0
        # Define observation points on edges parallel to northing
        easting, upward = tuple(
            c.ravel() for c in np.meshgrid([west, east], [bottom, top])
        )
        northing = np.full_like(
            easting, (south + north) / 2
        )  # put points in the center of the edge
        results = list(
            gravity_ee(e, n, u, *prism, density)
            for (e, n, u) in zip(easting, northing, upward)
        )
        assert np.isnan(results).all()
        # Define observation points on edges parallel to upward
        easting, northing = tuple(
            c.ravel() for c in np.meshgrid([west, east], [south, north])
        )
        upward = np.full_like(
            easting, (bottom + top) / 2
        )  # put points in the center of the edge
        results = list(
            gravity_ee(e, n, u, *prism, density)
            for (e, n, u) in zip(easting, northing, upward)
        )
        assert np.isnan(results).all()

    def test_gnn_on_edges(self, prism_boundaries):
        """
        Test if gravity_nn on edges perpendicular to northing is equal to NaN
        """
        # Define prism
        west, east, south, north, bottom, top = prism_boundaries
        prism = np.array([west, east, south, north, bottom, top])
        density = 1.0
        # Define observation points on edges parallel to easting
        northing, upward = tuple(
            c.ravel() for c in np.meshgrid([south, north], [bottom, top])
        )
        easting = np.full_like(
            northing, (west + east) / 2
        )  # put points in the center of the edge
        results = list(
            gravity_nn(e, n, u, *prism, density)
            for (e, n, u) in zip(easting, northing, upward)
        )
        assert np.isnan(results).all()
        # Define observation points on edges parallel to upward
        easting, northing = tuple(
            c.ravel() for c in np.meshgrid([west, east], [south, north])
        )
        upward = np.full_like(
            easting, (bottom + top) / 2
        )  # put points in the center of the edge
        results = list(
            gravity_nn(e, n, u, *prism, density)
            for (e, n, u) in zip(easting, northing, upward)
        )
        assert np.isnan(results).all()

    def test_guu_on_edges(self, prism_boundaries):
        """
        Test if gravity_uu on edges perpendicular to upward is equal to NaN
        """
        # Define prism
        west, east, south, north, bottom, top = prism_boundaries
        prism = np.array([west, east, south, north, bottom, top])
        density = 1.0
        # Define observation points on edges parallel to easting
        northing, upward = tuple(
            c.ravel() for c in np.meshgrid([south, north], [bottom, top])
        )
        easting = np.full_like(
            northing, (west + east) / 2
        )  # put points in the center of the edge
        results = list(
            gravity_uu(e, n, u, *prism, density)
            for (e, n, u) in zip(easting, northing, upward)
        )
        assert np.isnan(results).all()
        # Define observation points on edges parallel to northing
        easting, upward = tuple(
            c.ravel() for c in np.meshgrid([west, east], [bottom, top])
        )
        northing = np.full_like(
            easting, (south + north) / 2
        )  # put points in the center of the edge
        results = list(
            gravity_ee(e, n, u, *prism, density)
            for (e, n, u) in zip(easting, northing, upward)
        )
        assert np.isnan(results).all()

    def test_gee_faces_symmetry(self, prism_boundaries):
        """
        Test if g_ee returns the same values on the faces normal to easting
        """
        # Define prism
        west, east, south, north, bottom, top = prism_boundaries
        prism = np.array([west, east, south, north, bottom, top])
        density = 1.0
        # Define observation points in the two faces normal to easting
        # (without including the edges)
        northing = np.linspace(south, north, 21)[1:-1]
        upward = np.linspace(bottom, top, 21)[1:-1]
        northing, upward = tuple(c.ravel() for c in np.meshgrid(northing, upward))
        east_face = np.full_like(northing, east)
        west_face = np.full_like(northing, west)
        result_east_face = list(
            gravity_ee(e, n, u, *prism, density)
            for (e, n, u) in zip(east_face, northing, upward)
        )
        result_west_face = list(
            gravity_ee(e, n, u, *prism, density)
            for (e, n, u) in zip(west_face, northing, upward)
        )
        npt.assert_allclose(result_east_face, result_west_face)

    def test_gee_faces_limit(self, prism_boundaries):
        """
        Test if g_ee on faces normal to easting returns the limit when
        approaching from the outside of the prism
        """
        # Define prism
        west, east, south, north, bottom, top = prism_boundaries
        prism = np.array([west, east, south, north, bottom, top])
        density = 1.0
        # Compute g_ee on the east face and on a slightly displaced point
        northing = (south + north) / 2
        upward = (bottom + top) / 2
        g_ee_on_face = gravity_ee(east, northing, upward, *prism, density)
        g_ee_close_to_face = gravity_ee(east + 1e-3, northing, upward, *prism, density)
        # Compare the results
        assert np.sign(g_ee_on_face) == np.sign(g_ee_close_to_face)

    def test_gnn_faces_limit(self, prism_boundaries):
        """
        Test if g_nn on faces normal to northing returns the limit when
        approaching from the outside of the prism
        """
        # Define prism
        west, east, south, north, bottom, top = prism_boundaries
        prism = np.array([west, east, south, north, bottom, top])
        density = 1.0
        # Compute g_nn on the north face and on a slightly displaced point
        easting = (west + east) / 2
        upward = (bottom + top) / 2
        g_nn_on_face = gravity_ee(easting, north, upward, *prism, density)
        g_nn_close_to_face = gravity_ee(easting, north + 1e-3, upward, *prism, density)
        # Compare the results
        assert np.sign(g_nn_on_face) == np.sign(g_nn_close_to_face)

    def test_guu_faces_limit(self, prism_boundaries):
        """
        Test if g_uu on faces normal to northing returns the limit when
        approaching from the outside of the prism
        """
        # Define prism
        west, east, south, north, bottom, top = prism_boundaries
        prism = np.array([west, east, south, north, bottom, top])
        density = 1.0
        # Compute g_uu on the top face and on a slightly displaced point
        easting = (west + east) / 2
        northing = (south + north) / 2
        g_uu_on_face = gravity_ee(easting, northing, top, *prism, density)
        g_uu_close_to_face = gravity_ee(easting, northing, top + 1e-3, *prism, density)
        # Compare the results
        assert np.sign(g_uu_on_face) == np.sign(g_uu_close_to_face)

    def test_gnn_faces_symmetry(self, prism_boundaries):
        """
        Test if g_nn returns the same values on the faces normal to northing
        """
        # Define prism
        west, east, south, north, bottom, top = prism_boundaries
        prism = np.array([west, east, south, north, bottom, top])
        density = 1.0
        # Define observation points in the two faces normal to northing
        # (without including the edges)
        easting = np.linspace(west, east, 21)[1:-1]
        upward = np.linspace(bottom, top, 21)[1:-1]
        easting, upward = tuple(c.ravel() for c in np.meshgrid(easting, upward))
        south_face = np.full_like(easting, south)
        north_face = np.full_like(easting, north)
        result_north_face = list(
            gravity_nn(e, n, u, *prism, density)
            for (e, n, u) in zip(easting, north_face, upward)
        )
        result_south_face = list(
            gravity_nn(e, n, u, *prism, density)
            for (e, n, u) in zip(easting, south_face, upward)
        )
        npt.assert_allclose(result_north_face, result_south_face)

    def test_guu_faces_symmetry(self, prism_boundaries):
        """
        Test if g_uu returns the same values on the horizontal faces
        """
        # Define prism
        west, east, south, north, bottom, top = prism_boundaries
        prism = np.array([west, east, south, north, bottom, top])
        density = 1.0
        # Define observation points in the two horizontal faces
        # (without including the edges)
        easting = np.linspace(west, east, 21)[1:-1]
        northing = np.linspace(south, north, 21)[1:-1]
        easting, northing = tuple(c.ravel() for c in np.meshgrid(easting, northing))
        top_face = np.full_like(easting, top)
        bottom_face = np.full_like(easting, bottom)
        result_top_face = list(
            gravity_uu(e, n, u, *prism, density)
            for (e, n, u) in zip(easting, northing, top_face)
        )
        result_bottom_face = list(
            gravity_uu(e, n, u, *prism, density)
            for (e, n, u) in zip(easting, northing, bottom_face)
        )
        npt.assert_allclose(result_top_face, result_bottom_face)


class TestNonDiagonalTensorSingularities:
    """
    Test if non-diagonal tensor components behave as expected on their singular
    points

    Non-diagonal tensor components have singular points on:
    * prism vertices, and
    * prism edges perpendicular to the two tensor directions.

    In both cases, the forward modelling function should return ``np.nan``.
    """

    @pytest.fixture()
    def prism_boundaries(self):
        """
        Return the boundaries of the sample prism
        """
        west, east, south, north, bottom, top = -5.4, 10.1, 43.2, 79.5, -53.7, -44.3
        return west, east, south, north, bottom, top

    @pytest.mark.parametrize("function", (gravity_en, gravity_eu, gravity_nu))
    def test_on_vertices(self, prism_boundaries, function):
        """
        Test if non-diagonal tensor components on vertices are equal to NaN
        """
        west, east, south, north, bottom, top = prism_boundaries
        prism = np.array([west, east, south, north, bottom, top])
        density = 1.0
        coordinates = tuple(
            a.ravel() for a in np.meshgrid([west, east], [south, north], [bottom, top])
        )
        results = list(
            function(e, n, u, *prism, density) for (e, n, u) in zip(*coordinates)
        )
        assert np.isnan(results).all()

    def test_gen_edges(self, prism_boundaries):
        """
        Test if gravity_en returns np.nan on the vertical edges
        """
        # Define prism
        west, east, south, north, bottom, top = prism_boundaries
        prism = np.array([west, east, south, north, bottom, top])
        density = 1.0
        # Define observation points in the vertical edges of the prism
        easting, northing = tuple(
            c.ravel() for c in np.meshgrid([west, east], [south, north])
        )
        upward = np.full_like(
            easting, (bottom + top) / 2
        )  # put points in the center of the edge
        results = list(
            gravity_en(e, n, u, *prism, density)
            for (e, n, u) in zip(easting, northing, upward)
        )
        assert np.isnan(results).all()

    def test_geu_edges(self, prism_boundaries):
        """
        Test if gravity_eu returns np.nan on the edges parallel to northing
        """
        # Define prism
        west, east, south, north, bottom, top = prism_boundaries
        prism = np.array([west, east, south, north, bottom, top])
        density = 1.0
        # Define observation points on edges parallel to northing
        easting, upward = tuple(
            c.ravel() for c in np.meshgrid([west, east], [bottom, top])
        )
        northing = np.full_like(
            easting, (south + north) / 2
        )  # put points in the center of the edge
        results = list(
            gravity_eu(e, n, u, *prism, density)
            for (e, n, u) in zip(easting, northing, upward)
        )
        assert np.isnan(results).all()

    def test_gnu_edges(self, prism_boundaries):
        """
        Test if gravity_nu returns np.nan on the edges parallel to easting
        """
        # Define prism
        west, east, south, north, bottom, top = prism_boundaries
        prism = np.array([west, east, south, north, bottom, top])
        density = 1.0
        # Define observation points on edges parallel to easting
        northing, upward = tuple(
            c.ravel() for c in np.meshgrid([south, north], [bottom, top])
        )
        easting = np.full_like(
            northing, (west + east) / 2
        )  # put points in the center of the edge
        results = list(
            gravity_nu(e, n, u, *prism, density)
            for (e, n, u) in zip(easting, northing, upward)
        )
        assert np.isnan(results).all()
