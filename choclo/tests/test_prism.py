"""
Test kernel functions for rectangular prisms
"""
import pytest
import numpy as np
import numpy.testing as npt

from .. import prism_kernel_evaluation, kernel_prism_potential


@pytest.fixture(name="sample_prism_center")
def fixture_sample_prism_center():
    """
    Return the geometric center of the sample prism
    """
    return 30.5, 21.3, -43.5


@pytest.fixture(name="sample_prism_dimensions")
def fixture_sample_prism_dimensions():
    """
    Return the dimensions of the sample prism
    """
    return 10, 15, 20


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
    Test the symmetry of the kernel for the potential of a rectangular prism
    """

    scalers = [0.8, 1.0, 1.2]
    ids = ["inside", "surface", "outside"]

    @pytest.fixture(params=scalers, ids=ids)
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

    @pytest.fixture(params=scalers, ids=ids)
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

    @pytest.fixture(params=scalers, ids=ids)
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

    @pytest.fixture(params=scalers, ids=ids)
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

    @pytest.fixture(params=scalers, ids=ids)
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

    @pytest.fixture(params=scalers, ids=ids)
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

    @pytest.fixture(params=scalers, ids=ids)
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

    def test_vertices(self, coords_in_vertices, sample_prism):
        """
        Test if kernel for potential satisfies symmetry on vertices
        """
        # Compute the kernel on every observation point of the symmetry group
        kernel = list(
            prism_kernel_evaluation(e, n, u, sample_prism, kernel_prism_potential)
            for e, n, u in coords_in_vertices
        )
        npt.assert_allclose(kernel[0], kernel)

    def test_centers_of_easting_edges(
        self, coords_in_centers_of_easting_edges, sample_prism
    ):
        """
        Test if kernel for potential satisfies symmetry on centers of the edges
        parallel to the easting direction
        """
        # Compute the kernel on every observation point of the symmetry group
        kernel = list(
            prism_kernel_evaluation(e, n, u, sample_prism, kernel_prism_potential)
            for e, n, u in coords_in_centers_of_easting_edges
        )
        npt.assert_allclose(kernel[0], kernel)

    def test_centers_of_northing_edges(
        self, coords_in_centers_of_northing_edges, sample_prism
    ):
        """
        Test if kernel for potential satisfies symmetry on centers of the edges
        parallel to the northing direction
        """
        # Compute the kernel on every observation point of the symmetry group
        kernel = list(
            prism_kernel_evaluation(e, n, u, sample_prism, kernel_prism_potential)
            for e, n, u in coords_in_centers_of_northing_edges
        )
        npt.assert_allclose(kernel[0], kernel)

    def test_centers_of_upward_edges(
        self, coords_in_centers_of_upward_edges, sample_prism
    ):
        """
        Test if kernel for potential satisfies symmetry on centers of the edges
        parallel to the upward direction
        """
        # Compute the kernel on every observation point of the symmetry group
        kernel = list(
            prism_kernel_evaluation(e, n, u, sample_prism, kernel_prism_potential)
            for e, n, u in coords_in_centers_of_upward_edges
        )
        npt.assert_allclose(kernel[0], kernel)

    def test_centers_of_easting_faces(
        self, coords_in_centers_of_easting_faces, sample_prism
    ):
        """
        Test if kernel for potential satisfies symmetry on centers of the
        centers of the faces normal to the easting direction
        """
        # Compute the kernel on every observation point of the symmetry group
        kernel = list(
            prism_kernel_evaluation(e, n, u, sample_prism, kernel_prism_potential)
            for e, n, u in coords_in_centers_of_easting_faces
        )
        npt.assert_allclose(kernel[0], kernel)

    def test_centers_of_northing_faces(
        self, coords_in_centers_of_northing_faces, sample_prism
    ):
        """
        Test if kernel for potential satisfies symmetry on centers of the
        centers of the faces normal to the northing direction
        """
        # Compute the kernel on every observation point of the symmetry group
        kernel = list(
            prism_kernel_evaluation(e, n, u, sample_prism, kernel_prism_potential)
            for e, n, u in coords_in_centers_of_northing_faces
        )
        npt.assert_allclose(kernel[0], kernel)

    def test_centers_of_upward_faces(
        self, coords_in_centers_of_upward_faces, sample_prism
    ):
        """
        Test if kernel for potential satisfies symmetry on centers of the
        centers of the faces normal to the northing direction
        """
        # Compute the kernel on every observation point of the symmetry group
        kernel = list(
            prism_kernel_evaluation(e, n, u, sample_prism, kernel_prism_potential)
            for e, n, u in coords_in_centers_of_upward_faces
        )
        npt.assert_allclose(kernel[0], kernel)
