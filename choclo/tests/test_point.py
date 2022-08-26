# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Tests for kernel functions of point sources
"""
import numpy as np
import numpy.testing as npt
import pytest

from .._point import kernel_point_potential
from .utils import NUMBA_IS_DISABLED


@pytest.fixture(name="sample_point_source")
def fixture_sample_point_source():
    """
    Return a sample point source
    """
    return (3.7, 4.3, -5.8)


class TestSymmetryPotential:
    """
    Test the symmetry of the kernel for the potential of a point source
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

    def test_symmetry_on_sphere(self, sample_point_source):
        """
        Test the symmetry of the potential in points of a sphere
        """
        radius = 3.5
        observation_points = self.build_points_in_sphere(radius, sample_point_source)
        kernel_potential = [
            kernel_point_potential(*coords, *sample_point_source)
            for coords in zip(*observation_points)
        ]
        npt.assert_allclose(kernel_potential[0], kernel_potential)

    def test_potential_between_two_spheres(self, sample_point_source):
        """
        Test the potential between observation points in two spheres
        """
        radius_1, radius_2 = 3.5, 8.7
        sphere_1 = self.build_points_in_sphere(radius_1, sample_point_source)
        sphere_2 = self.build_points_in_sphere(radius_2, sample_point_source)
        kernel_potential_1 = np.array(
            [
                kernel_point_potential(*coords, *sample_point_source)
                for coords in zip(*sphere_1)
            ]
        )
        kernel_potential_2 = np.array(
            [
                kernel_point_potential(*coords, *sample_point_source)
                for coords in zip(*sphere_2)
            ]
        )
        npt.assert_allclose(
            kernel_potential_1, radius_2 / radius_1 * kernel_potential_2
        )

    @pytest.mark.skipif(not NUMBA_IS_DISABLED, reason="Numba is not disabled")
    def test_infinite_potential(self):
        """
        Test if we get an infinite kernel if the computation point is in the
        same location as the observation point.

        This test should be run only with Numba disabled.
        """
        point = (4.6, -8.9, -50.3)
        assert np.isinf(kernel_point_potential(*point, *point))

    @pytest.mark.skipif(NUMBA_IS_DISABLED, reason="Numba is disabled")
    def test_division_by_zero(self):
        """
        Test if we get a division by zero if the computation point is in the
        same location as the observation point.

        This test should be run only with Numba enabled.
        """
        point = (4.6, -8.9, -50.3)
        with pytest.raises(ZeroDivisionError):
            kernel_point_potential(*point, *point)
