# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Functions to compute Euclidean distance in different coordinate systems.
"""

import numba
import numpy as np


@numba.jit(nopython=True)
def distance_cartesian(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q
):
    r"""
    Euclidean distance between two points given in Cartesian coordinates.

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


@numba.jit(nopython=True)
def distance_spherical(
    longitude_p, latitude_p, radius_p, longitude_q, latitude_q, radius_q
):
    r"""
    Euclidean distance between two points in spherical coordinates.

    .. important::

        All angles must be in degrees and radii in meters.

    Parameters
    ----------
    longitude_p, latitude_p, radius_p : float
        Longitude, geocentric latitude, and radial coordinates of point
        :math:`\mathbf{p}`. Longitude and latitude must be in degrees. Radial
        coordinate should be in meters.
    longitude_q, latitude_q, radius_q : float
        Longitude, geocentric latitude, and radial coordinates of point
        :math:`\mathbf{q}`. Longitude and latitude must be in degrees. Radial
        coordinate should be in meters.

    Returns
    -------
    distance : float
        Euclidean distance between point :math:`\mathbf{p}` and point
        :math:`\mathbf{q}` in meters.

    Notes
    -----
    Consider two points :math:`\mathbf{p} = (\lambda_p, \phi_p, r_p)` and
    :math:`\mathbf{q} = (\lambda_q, \phi_q, r_q)`. Both are defined on a
    geocentric spherical coordinate system (:math:`\lambda` is the longitude,
    :math:`\phi` is the geocentric latitude, and :math:`r` is the radius). The
    Euclidean (L2) distance between them is:

    .. math::

        d = \sqrt{ (r_p - r_q) ^ 2 + 2 r_p r_q (1 - \cos\psi)},

    where

    .. math::

        \cos\psi = \sin\phi_p \sin\phi_q
            + \cos\phi_p \cos\phi_q \cos(\lambda_p - \lambda_q).
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


@numba.jit(nopython=True)
def distance_spherical_core(
    longitude_p, cosphi_p, sinphi_p, radius_p, longitude_q, cosphi_q, sinphi_q, radius_q
):
    r"""
    Core computation of distance between two points in spherical coordinates.

    Computes the Euclidean distance between two points defined in spherical
    coordinates given precomputed quantities related to the coordinates of both
    points: the longitude in radians, the sine and cosine of the geocentric
    latitude, and the radius in meters. Precomputing this quantities may save
    computation time in some cases.

    .. important::

        All longitudinal angles must be in radians and radii in meters.

    Parameters
    ----------
    longitude_p : float
        Longitude coordinate of the first point. Must be in radians.
    cosphi_p : float
        Cosine of the geocentric latitude coordinate of the first point.
    sinphi_p : float
        Sine of the geocentric latitude coordinate of the first point.
    radius_p : float
        Radial coordinate of the first point.
    longitude_q : float
        Longitude coordinate of the second point. Must be in radians.
    cosphi_q : float
        Cosine of the geocentric latitude coordinate of the second point.
    sinphi_q : float
        Sine of the geocentric latitude coordinate of the second point.
    radius_q : float
        Radial coordinate of the second point.

    Returns
    -------
    distance : float
        Euclidean distance between point :math:`\mathbf{p}` and point
        :math:`\mathbf{q}` in meters.
    cospsi : float
        Cosine of the psi angle (see below).
    coslambda : float
        Cosine of the difference between the longitudes of both points.

    Notes
    -----
    Consider two points :math:`\mathbf{p} = (\lambda_p, \phi_p, r_p)` and
    :math:`\mathbf{q} = (\lambda_q, \phi_q, r_q)`. Both are defined on a
    geocentric spherical coordinate system (:math:`\lambda` is the longitude,
    :math:`\phi` is the geocentric latitude, and :math:`r` is the radius). The
    Euclidean (L2) distance between them is:

    .. math::

        d = \sqrt{ (r_p - r_q) ^ 2 + 2 r_p r_q (1 - \cos\psi)},

    where

    .. math::

        \cos\psi = \sin\phi_p \sin\phi_q
            + \cos\phi_p \cos\phi_q \cos(\lambda_p - \lambda_q).
    """
    coslambda = np.cos(longitude_q - longitude_p)
    cospsi = sinphi_q * sinphi_p + cosphi_q * cosphi_p * coslambda
    distance = np.sqrt(
        (radius_p - radius_q) ** 2 + 2 * radius_p * radius_q * (1 - cospsi)
    )
    return distance, cospsi, coslambda


@numba.jit(nopython=True)
def distance_geodetic(
    longitude_p,
    latitude_p,
    height_p,
    longitude_q,
    latitude_q,
    height_q,
    semimajor_axis,
    flattening,
):
    r"""
    Euclidean distance between two points in geodetic coordinates.

    Computes the Euclidean distance between two points given in geodetic
    coordinates using the closed-form formula given by [Vajda2004]_.

    .. important::

        All angles must be in degrees and heights in meters.

    .. hint::

        Use :mod:`boule` to obtain the necessary ellipsoid parameters.

    Parameters
    ----------
    longitude_p, latitude_p, height_p : float
        Longitude, geodetic latitude, and height above the ellipsoid coordinates
        of point :math:`\mathbf{p}`. Longitude and latitude must be in degrees,
        height should be in meters.
    longitude_q, latitude_q, height_q : float
        Longitude, geodetic latitude, and height above the ellipsoid coordinates
        of point :math:`\mathbf{q}`. Longitude and latitude must be in degrees,
        height should be in meters.
    semimajor_axis : float
        The semimajor (equatorial) axis of the reference ellipsoid in meters.
    flattening : float
        The flattening of the ellipsoid :math:`f = (a - b)/a`, in which
        :math:`a` is the semimajor axis and :math:`b` is the semiminor axis of
        the ellipsoid.

    Returns
    -------
    distance : float
        Euclidean distance between point :math:`\mathbf{p}` and point
        :math:`\mathbf{q}` in meters.
    """
    # Convert angles to radians
    longitude_p, latitude_p = np.radians(longitude_p), np.radians(latitude_p)
    longitude_q, latitude_q = np.radians(longitude_q), np.radians(latitude_q)
    # Compute trigonometric quantities
    cosphi_p = np.cos(latitude_p)
    sinphi_p = np.sin(latitude_p)
    cosphi_q = np.cos(latitude_q)
    sinphi_q = np.sin(latitude_q)
    # Calculate the ellipsoid's first eccentricity
    eccentricity = np.sqrt(2 * flattening - flattening**2)
    # Compute prime vertical radii for both points
    prime_radius_p = semimajor_axis / np.sqrt(1 - eccentricity**2 * sinphi_p**2)
    prime_radius_q = semimajor_axis / np.sqrt(1 - eccentricity**2 * sinphi_q**2)
    # Compute the Euclidean distance using the close-form formula
    distance = distance_geodetic_core(
        longitude_p,
        cosphi_p,
        sinphi_p,
        height_p,
        longitude_q,
        cosphi_q,
        sinphi_q,
        height_q,
        prime_radius_p,
        prime_radius_q,
        eccentricity,
    )
    return distance


@numba.jit(nopython=True)
def distance_geodetic_core(
    longitude_p,
    cosphi_p,
    sinphi_p,
    height_p,
    longitude_q,
    cosphi_q,
    sinphi_q,
    height_q,
    prime_radius_p,
    prime_radius_q,
    eccentricity,
):
    r"""
    Core computation of distance between two points in geodetic coordinates.

    Computes the Euclidean distance between two points defined in geodetic
    coordinates given precomputed quantities related to the coordinates of
    both points: the longitude in radians, the sine and cosine of the geodetic
    latitude, and geometric properties of the ellipsoid. Precomputing this
    quantities may save computation time in some cases.

    .. important::

        All longitudinal angles must be in radians and heights in meters.

    .. hint::

        Use :mod:`boule` to obtain the necessary ellipsoid parameters.

    Parameters
    ----------
    longitude_p : float
        Longitude coordinate of the first point. Must be in radians.
    cosphi_p : float
        Cosine of the geodetic latitude coordinate of the first point.
    sinphi_p : float
        Sine of the geodetic latitude coordinate of the first point.
    height_p : float
        Height above the ellipsoid of the first point in meters.
    longitude_q : float
        Longitude coordinate of the second point. Must be in radians.
    cosphi_q : float
        Cosine of the geodetic latitude coordinate of the second point.
    sinphi_q : float
        Sine of the geodetic latitude coordinate of the second point.
    height_q : float
        Height above the ellipsoid of the second point in meters.
    prime_radius_p : float
        Prime vertical radius for the latitude of the first point
        :math:`N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2(\phi)}}` in which
        :math:`a` is the semimajor axis and :math:`e` is the eccentricity of
        the ellipsoid. Must be in meters.
    prime_radius_q : float
        Prime vertical radius for the latitude of the second point. Must be
        in meters.
    eccentricity : float
        The first eccentricity of the ellipsoid :math:`e = \sqrt{2f - f^2}`, in
        which :math:`f` is the flattening.

    Returns
    -------
    distance : float
        Euclidean distance between point :math:`\mathbf{p}` and point
        :math:`\mathbf{q}` in meters.
    """
    upward_sum_p = prime_radius_p + height_p
    upward_sum_q = prime_radius_q + height_q
    coslambda = np.cos(longitude_q - longitude_p)
    distance = np.sqrt(
        upward_sum_p**2 * cosphi_p**2
        + upward_sum_q**2 * cosphi_q**2
        - 2 * upward_sum_q * upward_sum_p * cosphi_q * cosphi_p * coslambda
        + (upward_sum_p - eccentricity**2 * prime_radius_p) ** 2 * sinphi_p**2
        + (upward_sum_q - eccentricity**2 * prime_radius_q) ** 2 * sinphi_q**2
        - (
            2
            * (upward_sum_p - eccentricity**2 * prime_radius_p)
            * (upward_sum_q - eccentricity**2 * prime_radius_q)
            * sinphi_q
            * sinphi_p
        )
    )
    return distance
