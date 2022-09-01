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

    @pytest.fixture
    def coords_in_vertices(self, sample_prism):
        """
        Return observation points located in the vertices of the prism
        """
        west, east, south, north, bottom, top = sample_prism[:]
        vertices = list(
            [easting, northing, upward]
            for easting in (west, east)
            for northing in (south, north)
            for upward in (bottom, top)
        )
        return vertices

    @pytest.fixture
    def coords_in_centers_of_easting_edges(
        self, sample_prism_center, sample_prism_dimensions
    ):
        """
        Return observation points located in the center of the prism edges
        parallel to easting axis
        """
        # Get the coordinates of the sample prism center
        easting, northing, upward = sample_prism_center
        # Get the dimensions of the sample prism
        d_easting, _, d_upward = sample_prism_dimensions
        # Get the points in the symmetry group
        edges_easting = [
            [easting - d_easting / 2, northing, upward - d_upward / 2],
            [easting - d_easting / 2, northing, upward + d_upward / 2],
            [easting + d_easting / 2, northing, upward - d_upward / 2],
            [easting + d_easting / 2, northing, upward + d_upward / 2],
        ]
        return edges_easting

    @pytest.fixture
    def coords_in_centers_of_northing_edges(
        self, sample_prism_center, sample_prism_dimensions
    ):
        """
        Return observation points located in the center of the prism edges
        parallel to northing axis
        """
        # Get the coordinates of the sample prism center
        easting, northing, upward = sample_prism_center
        # Get the dimensions of the sample prism
        _, d_northing, d_upward = sample_prism_dimensions
        # Get the points in the symmetry group
        edges_northing = [
            [easting, northing - d_northing / 2, upward - d_upward / 2],
            [easting, northing - d_northing / 2, upward + d_upward / 2],
            [easting, northing + d_northing / 2, upward - d_upward / 2],
            [easting, northing + d_northing / 2, upward + d_upward / 2],
        ]
        return edges_northing

    @pytest.fixture
    def coords_in_centers_of_upward_edges(
        self, sample_prism_center, sample_prism_dimensions
    ):
        """
        Return observation points located in the center of the prism edges
        parallel to upward axis
        """
        # Get the coordinates of the sample prism center
        easting, northing, upward = sample_prism_center
        # Get the dimensions of the sample prism
        d_easting, d_northing, _ = sample_prism_dimensions
        # Get the points in the symmetry group
        edges_upward = [
            [easting - d_easting / 2, northing - d_northing / 2, upward],
            [easting + d_easting / 2, northing - d_northing / 2, upward],
            [easting - d_easting / 2, northing + d_northing / 2, upward],
            [easting + d_easting / 2, northing + d_northing / 2, upward],
        ]
        return edges_upward

    @pytest.fixture
    def coords_in_centers_of_easting_faces(
        self, sample_prism_center, sample_prism_dimensions
    ):
        """
        Return observation points located in the center of the faces normal to
        the easting axis
        """
        # Get the coordinates of the sample prism center
        easting, northing, upward = sample_prism_center
        # Get the dimensions of the sample prism
        d_easting, _, _ = sample_prism_dimensions
        # Get the points in the symmetry group
        faces_normal_to_easting = [
            [easting - d_easting / 2, northing, upward],
            [easting + d_easting / 2, northing, upward],
        ]
        return faces_normal_to_easting

    @pytest.fixture
    def coords_in_centers_of_northing_faces(
        self, sample_prism_center, sample_prism_dimensions
    ):
        """
        Return observation points located in the center of the faces normal to
        the northing axis
        """
        # Get the coordinates of the sample prism center
        easting, northing, upward = sample_prism_center
        # Get the dimensions of the sample prism
        _, d_northing, _ = sample_prism_dimensions
        # Get the points in the symmetry group
        faces_normal_to_northing = [
            [easting, northing - d_northing / 2, upward],
            [easting, northing + d_northing / 2, upward],
        ]
        return faces_normal_to_northing

    @pytest.fixture
    def coords_in_centers_of_upward_faces(
        self, sample_prism_center, sample_prism_dimensions
    ):
        """
        Return observation points located in the center of the faces normal to
        the northing axis
        """
        # Get the coordinates of the sample prism center
        easting, northing, upward = sample_prism_center
        # Get the dimensions of the sample prism
        _, _, d_upward = sample_prism_dimensions
        # Get the points in the symmetry group
        faces_normal_to_upward = [
            [easting, northing, upward - d_upward / 2],
            [easting, northing, upward + d_upward / 2],
        ]
        return faces_normal_to_upward

    @pytest.mark.parametrize(
        "coords",
        (
            "coords_in_vertices",
            "coords_in_centers_of_easting_edges",
            "coords_in_centers_of_northing_edges",
            "coords_in_centers_of_upward_edges",
            "coords_in_centers_of_easting_faces",
            "coords_in_centers_of_northing_faces",
            "coords_in_centers_of_upward_faces",
        ),
    )
    def test_symmetry(self, coords, sample_prism, request):
        """
        Test if kernel function for  potential field satisfies symmetry

        The test is run over a set of symmetry groups. Each symmetry group is
        composed by a set of observation points where the kernel function must
        have the same value.
        """
        # Get the coordinates of the symmetry group through its corresponding
        # fixture (use pytest request fixture to do so)
        coords = request.getfixturevalue(coords)
        # Compute the kernel on every observation point of the symmetry group
        kernel = list(
            prism_kernel_evaluation(
                easting, northing, upward, sample_prism, kernel_prism_potential
            )
            for easting, northing, upward in coords
        )
        npt.assert_allclose(kernel[0], kernel)
