"""
Test gravity forward modelling functions for point sources
"""
import pytest
import numpy as np
import numpy.testing as npt

from ..point import gravity_pot
from .utils import NUMBA_IS_DISABLED


@pytest.fixture(name="sample_point_source")
def fixture_sample_point_source():
    """
    Return a sample point source
    """
    return (3.7, 4.3, -5.8)


@pytest.fixture(name="sample_mass")
def fixture_sample_mass():
    """
    Return the mass for the sample point source
    """
    return 9e4


@pytest.fixture(name="sample_coordinate")
def fixture_sample_coordinate():
    """
    Define a sample observation point
    """
    return (16.7, 13.2, 7.8)


class TestSymmetryPotential:
    """
    Test the symmetry of gravity potential due to a point source
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

    def test_symmetry_on_sphere(self, sample_point_source, sample_mass):
        """
        Test the symmetry of the gravity potential in points of a sphere
        """
        radius = 3.5
        observation_points = self.build_points_in_sphere(radius, sample_point_source)
        # Evaluate gravity_pot on every observation point
        potential = tuple(
            gravity_pot(*coords, *sample_point_source, sample_mass)
            for coords in zip(*observation_points)
        )
        # Check if all values are equal
        npt.assert_allclose(potential[0], potential)

    def test_potential_between_two_spheres(self, sample_point_source, sample_mass):
        """
        Test the gravity potential between observation points in two spheres
        """
        radius_1, radius_2 = 3.5, 8.7
        sphere_1 = self.build_points_in_sphere(radius_1, sample_point_source)
        sphere_2 = self.build_points_in_sphere(radius_2, sample_point_source)
        # Evaluate kernel_pot on every observation point
        potential_1 = gravity_pot(*sphere_1, *sample_point_source, sample_mass)
        potential_2 = gravity_pot(*sphere_2, *sample_point_source, sample_mass)
        # Check if all values are equal
        npt.assert_allclose(potential_1, radius_2 / radius_1 * potential_2)

    @pytest.mark.skipif(not NUMBA_IS_DISABLED, reason="Numba is not disabled")
    def test_infinite_potential(self, sample_mass):
        """
        Test if we get an infinite kernel if the computation point is in the
        same location as the observation point.

        This test should be run only with Numba disabled.
        """
        point = (4.6, -8.9, -50.3)
        assert np.isinf(gravity_pot(*point, *point, sample_mass))

    @pytest.mark.skipif(NUMBA_IS_DISABLED, reason="Numba is disabled")
    def test_division_by_zero(self, sample_mass):
        """
        Test if we get a division by zero if the computation point is in the
        same location as the observation point.

        This test should be run only with Numba enabled.
        """
        point = (4.6, -8.9, -50.3)
        with pytest.raises(ZeroDivisionError):
            gravity_pot(*point, *point, sample_mass)
