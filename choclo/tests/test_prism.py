"""
Test forward modelling functions for rectangular prisms
"""
import pytest
import numpy as np
import numpy.testing as npt

from ..prism import gravity_pot, gravity_u


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
            gravity_pot(e, n, u, sample_prism, sample_density)
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
            gravity_pot(e, n, u, sample_prism, sample_density)
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
            gravity_pot(e, n, u, sample_prism, sample_density)
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
            gravity_pot(e, n, u, sample_prism, sample_density)
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
            gravity_pot(e, n, u, sample_prism, sample_density)
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
            gravity_pot(e, n, u, sample_prism, sample_density)
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
            gravity_pot(e, n, u, sample_prism, sample_density)
            for e, n, u in coords_in_centers_of_upward_faces
        )
        npt.assert_allclose(potential[0], potential)


class TestSymmetryGravityU:
    """
    Test the symmetry of gravity_u of a rectangular prism
    """

    scalers = {
        "inside": 0.8,
        "surface": 1.0,
        "outside": 1.2,
    }

    @pytest.fixture(params=scalers.values(), ids=scalers.keys())
    def coords_in_easting_northing_plane(
        self, sample_prism_center, sample_prism_dimensions, request
    ):
        """
        Return observation points located in the easting-northing plane that
        pass along the center of the prism.
        """
        # Get the coordinates of the sample prism center
        easting, northing, upward = sample_prism_center
        # Get the dimensions of the sample prism
        d_easting, d_northing, d_upward = sample_prism_dimensions
        # Get scaler
        scaler = request.param
        # Build the points
        points = list(
            [
                easting + i * scaler * d_easting / 2,
                northing + j * scaler * d_northing / 2,
                upward,
            ]
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            if i != 0 and j != 0
        )
        return points

    @pytest.fixture(params=scalers.values(), ids=scalers.keys())
    def coords_in_top_and_bottom_planes(
        self, sample_prism_center, sample_prism_dimensions, request
    ):
        """
        Return observation points located in top and bottom easting-northing
        planes symmetrical to the prism center.
        """
        # Get the coordinates of the sample prism center
        easting, northing, upward = sample_prism_center
        # Get the dimensions of the sample prism
        d_easting, d_northing, d_upward = sample_prism_dimensions
        # Get scaler
        scaler = request.param
        # Build the bottom and top points
        bottom_points = list(
            [
                easting + i * scaler * d_easting / 2,
                northing + j * scaler * d_northing / 2,
                upward - scaler * d_upward / 2,
            ]
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            if i != 0 and j != 0
        )
        top_points = list(
            [
                easting + i * scaler * d_easting / 2,
                northing + j * scaler * d_northing / 2,
                upward + scaler * d_upward / 2,
            ]
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            if i != 0 and j != 0
        )
        return bottom_points, top_points

    def test_easting_northing_plane(
        self, coords_in_easting_northing_plane, sample_prism, sample_density
    ):
        """
        Test if gravity_u is null in points of the easting-northing plane that
        passes through its center.
        """
        # Compute gravity_u on every observation point of the symmetry group
        g_u = np.array(
            list(
                gravity_u(e, n, u, sample_prism, sample_density)
                for e, n, u in coords_in_easting_northing_plane
            )
        )
        assert (g_u < 1e-32).all()

    def test_bottom_top_planes(
        self, coords_in_top_and_bottom_planes, sample_prism, sample_density
    ):
        """
        Test if gravity_u is opposite in points of the top and bottom
        easting-northing planes.
        """
        bottom, top = coords_in_top_and_bottom_planes
        # Compute gravity_u on every observation point of the two planes
        g_u_bottom = np.array(
            list(gravity_u(e, n, u, sample_prism, sample_density) for e, n, u in bottom)
        )
        g_u_top = np.array(
            list(gravity_u(e, n, u, sample_prism, sample_density) for e, n, u in top)
        )
        npt.assert_allclose(g_u_bottom, -g_u_top)

    def test_sign(self, sample_prism_center, sample_prism, sample_density):
        """
        Test that gravity_u has the correct sign on top and bottom points
        """
        d_upward = sample_prism[5] - sample_prism[4]
        # Define an observation point on top of the prism
        point = (
            sample_prism_center[0],
            sample_prism_center[1],
            sample_prism_center[2] + d_upward,
        )
        # Compute gravity_u
        g_u = gravity_u(*point, sample_prism, sample_density)
        # Check if g_u has the opposite sign as its density (the acceleration
        # will point downwards for a positive density, therefore negative
        # upward component)
        assert np.sign(g_u) != np.sign(sample_density)


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
    def finite_diff_gravity_u(self, sample_coordinate, sample_prism, sample_density):
        """
        Compute gravity_u through finite differences of the gravity_pot
        """
        easting_p, northing_p, upward_p = sample_coordinate
        bottom, top = sample_prism[-2:]
        # Compute a small increment in the easting coordinate
        center_upward = (top + bottom) / 2  # upward coord of the prism center
        d_upward = self.delta_percentage * (upward_p - center_upward)
        # Compute shifted coordinate
        shifted_coordinate = (easting_p, northing_p, upward_p + d_upward)
        # Calculate g_u through finite differences
        g_u = (
            gravity_pot(*shifted_coordinate, *sample_prism, sample_density)
            - gravity_pot(*sample_coordinate, *sample_prism, sample_density)
        ) / d_upward
        return g_u