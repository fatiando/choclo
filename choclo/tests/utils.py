# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Define utility functions for running the tests
"""
import os

import numpy as np

# Determine if Numba just-in-time compilation is disabled
NUMBA_IS_DISABLED = os.environ.get("NUMBA_DISABLE_JIT", default="0") != "0"


def spherical_to_geocentric_cartesian(longitude, latitude, radius):
    """
    Convert spherical coordinates into geocentric Cartesian coordinates

    Parameters
    ----------
    longitude : float
        Longitudinal coordinate (in degrees).
    latitude : float
        Latitudinal coordinate (in degrees).
    radius : float
        Radial coordinates (in meters).

    Returns
    -------
    x, y, z : floats
        Geocentric Cartesian coordinates of the passed point.
    """
    longitude, latitude = np.radians(longitude), np.radians(latitude)
    x = radius * np.cos(longitude) * np.cos(latitude)
    y = radius * np.sin(longitude) * np.cos(latitude)
    z = radius * np.sin(latitude)
    return x, y, z


def dumb_spherical_distance(point_p, point_q):
    """
    Dumb calculation of distance between two points in spherical coordinates

    This function converts the spherical coordinates of each point to
    geocentric Cartesian coordinates and then uses a Cartesian Euclidean
    distance calculation.

    .. warning:

        This implementation is less optimal than the one we have in
        :func:`choclo.distance_spherical`. It exists only for testing purposes.

    .. important:

        All angles must be in degrees and radii in meters.

    Parameters
    ----------
    point_p : tuple or 1d-array
        Tuple or array containing the coordinates of the first point in the
        following order: (``longitude``, ``latitude`` and ``radius``).
        Both ``longitude`` and ``latitude`` must be in degrees, while
        ``radius`` in meters.
    point_q : tuple or 1d-array
        Tuple or array containing the coordinates of the second point in the
        following order: (``longitude``, ``latitude`` and ``radius``).
        Both ``longitude`` and ``latitude`` must be in degrees, while
        ``radius`` in meters.

    Returns
    -------
    distance : float
        Euclidean distance between ``point_p`` and ``point_q``.
    """
    # Convert points to geocentric Cartesian coordinates
    point_p = spherical_to_geocentric_cartesian(*point_p)
    point_q = spherical_to_geocentric_cartesian(*point_q)
    # Convert points to arrays
    point_p, point_q = np.asarray(point_p), np.asarray(point_q)
    # Calculate Euclidean Cartesian distance between these two
    return np.sqrt(np.sum((point_p - point_q) ** 2))
