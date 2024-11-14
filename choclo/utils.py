# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Utility functions for the kernel and forward modelling functions
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def distance_cartesian(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q
):
    r"""
    Euclidean distance between two points given in Cartesian coordinates

    .. warning::

        All coordinates should be in the same units.

    Parameters
    ----------
    easting_p, northing_p, upward_p : float
        Easting, northing and upward coordinates of point :math:`\mathbf{p}`.
    easting_q, northing_q, upward_q : float
        Easting, northing and upward coordinates of point :math:`\mathbf{q}`.

    Returns
    -------
    distance : float
        Euclidean distance between ``point_p`` and ``point_q``.

    Notes
    -----
    Given two points :math:`\mathbf{p} = (x_p, y_p, z_p)` and
    :math:`\mathbf{q} = (x_q, y_q, z_q)` defined in a Cartesian coordinate
    system :math:`(x, y, z)`, return the Euclidean (L2) distance between
    them:

    .. math::

        d = \sqrt{(x_p - x_q)^2 + (y_p - y_q)^2 + (z_p - z_q)^2}

    """
    distance = np.sqrt(
        (easting_p - easting_q) ** 2
        + (northing_p - northing_q) ** 2
        + (upward_p - upward_q) ** 2
    )
    return distance


@jit(nopython=True)
def distance_spherical(
    longitude_p, latitude_p, radius_p, longitude_q, latitude_q, radius_q
):
    r"""
    Euclidean distance between two points in spherical coordinates

    .. important::

        All angles must be in degrees and radii in meters.

    Parameters
    ----------
    longitude_p, latitude_p, radius_p : float
        Longitude, latitude and radial coordinates of point :math:`\mathbf{p}`.
        Longitude and latitude must be in degrees. Radial coordinate should be
        in meters.
    longitude_q, latitude_q, radius_q : float
        Longitude, latitude and radial coordinates of point :math:`\mathbf{q}`.
        Longitude and latitude must be in degrees. Radial coordinate should be
        in meters.

    Returns
    -------
    distance : float
        Euclidean distance between point :math:`\mathbf{p}` and point
        :math:`\mathbf{q}`.

    Notes
    -----
    Given two points :math:`\mathbf{p} = (\lambda_p, \phi_p, r_p)` and
    :math:`\mathbf{q} = (\lambda_q, \phi_q, r_q)` defined in a spherical
    coordinate system :math:`(\lambda, \phi, r)`, return the Euclidean (L2)
    distance between them:

    .. math::

        d = \sqrt{ (r_p - r_q) ^ 2 + 2 r_p r_q (1 - \cos\psi)}

    where

    .. math::

        \cos\psi =
        \sin\phi_p \sin\phi_q
        + \cos\phi_p \cos\phi_q \cos(\lambda_p - \lambda_q)

    and :math:`\lambda` is the longitude angle, :math:`\phi` the spherical
    latitude angle an :math:`r` is the radial coordinate.
    """
    # Convert angles to radians
    longitude_p, latitude_p = np.radians(longitude_p), np.radians(latitude_p)
    longitude_q, latitude_q = np.radians(longitude_q), np.radians(latitude_q)
    # Compute trigonometric quantities
    cosphi_p = np.cos(latitude_p)
    sinphi_p = np.sin(latitude_p)
    cosphi_q = np.cos(latitude_q)
    sinphi_q = np.sin(latitude_q)
    distance, _, _ = distance_spherical_core(
        longitude_p,
        cosphi_p,
        sinphi_p,
        radius_p,
        longitude_q,
        cosphi_q,
        sinphi_q,
        radius_q,
    )
    return distance


@jit(nopython=True)
def distance_spherical_core(
    longitude_p, cosphi_p, sinphi_p, radius_p, longitude_q, cosphi_q, sinphi_q, radius_q
):
    r"""
    Core computation of distance between two points in spherical coordinates

    .. important::

        All longitudinal angles must be in degrees.

    It computes the Euclidean distance between two points defined in spherical
    coordinates given precomputed quantities related to the coordinates of both
    points: the ``longitude`` in radians, the sine and cosine of the
    ``latitude``, and the ``radius`` in meters. Precomputing this quantities
    may save computation time on some cases.

    Parameters
    ----------
    longitude_p : float
        Longitude coordinate of the first point. Must be in radians.
    cosphi_p : float
        Cosine of the latitude coordinate of the first point.
    sinphi_p : float
        Sine of the latitude coordinate of the first point.
    radius_p : float
        Radial coordinate of the first point.
    longitude_q : float
        Longitude coordinate of the second point. Must be in radians.
    cosphi_q : float
        Cosine of the latitude coordinate of the second point.
    sinphi_q : float
        Sine of the latitude coordinate of the second point.
    radius_q : float
        Radial coordinate of the second point.

    Returns
    -------
    distance : float
        Distance between the two points.
    cospsi : float
        Cosine of the psi angle.
    coslambda : float
        Cosine of the diference between the longitudes of both points.

    Notes
    -----
    Given two points :math:`\mathbf{p} = (\lambda_p, \phi_p, r_p)` and
    :math:`\mathbf{q} = (\lambda_q, \phi_q, r_q)` defined in a spherical
    coordinate system :math:`(\lambda, \phi, r)`, return the Euclidean (L2)
    distance between them:

    .. math::

        d = \sqrt{ (r_p - r_q) ^ 2 + 2 r_p r_q (1 - \cos\psi)},

    the cosine of the :math:`\psi` angle:

    .. math::

        \cos\psi =
        \sin\phi_p \sin\phi_q
        + \cos\phi_p \cos\phi_q \cos(\lambda_p - \lambda_q)

    and the cosine of the difference between the longitude angles of the
    observation and source points:

    .. math::

        \cos(\lambda_p - \lambda_q),

    where :math:`\lambda` is the longitude angle, :math:`\phi` the spherical
    latitude angle an :math:`r` is the radial coordinate.
    """
    coslambda = np.cos(longitude_q - longitude_p)
    cospsi = sinphi_q * sinphi_p + cosphi_q * cosphi_p * coslambda
    distance = np.sqrt(
        (radius_p - radius_q) ** 2 + 2 * radius_p * radius_q * (1 - cospsi)
    )
    return distance, cospsi, coslambda
