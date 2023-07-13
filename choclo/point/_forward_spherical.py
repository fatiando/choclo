# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Forward modelling function for point sources
"""
import numpy as np
from numba import jit

from ..constants import GRAVITATIONAL_CONST
from ..utils import distance_spherical_core


@jit(nopython=True)
def gravity_pot_spherical(
    longitude_p,
    cosphi_p,
    sinphi_p,
    radius_p,
    longitude_q,
    cosphi_q,
    sinphi_q,
    radius_q,
    mass,
):
    """
    Kernel function for potential gravitational field in spherical coordinates
    """
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
    kernel = 1 / distance
    return GRAVITATIONAL_CONST * mass * kernel


#  Acceleration components
#  -------------------


@jit(nopython=True)
def gravity_u_spherical(
    longitude_p,
    cosphi_p,
    sinphi_p,
    radius_p,
    longitude_q,
    cosphi_q,
    sinphi_q,
    radius_q,
    mass,
):
    """
    Upward component of the gravitational acceleration due to a point source

    Returns the upward component of the gravitational acceleration produced by
    a single point source on a single computation point

    Use spherical coordinates
    """
    distance, cospsi, _ = distance_spherical_core(
        longitude_p,
        cosphi_p,
        sinphi_p,
        radius_p,
        longitude_q,
        cosphi_q,
        sinphi_q,
        radius_q,
    )
    delta_z = radius_p - radius_q * cospsi
    kernel = -delta_z / distance**3
    return GRAVITATIONAL_CONST * mass * kernel


@jit(nopython=True)
def gravity_e_spherical(
    longitude_p,
    cosphi_p,
    sinphi_p,
    radius_p,
    longitude_q,
    cosphi_q,
    sinphi_q,
    radius_q,
    mass,
):
    """
    Easting component of the gravitational acceleration due to a point source

    Returns the upward component of the gravitational acceleration produced by
    a single point source on a single computation point

    Use spherical coordinates
    """
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
    delta_e = radius_q * cosphi_q * np.sin(longitude_q - longitude_p)
    kernel = delta_e / distance**3
    return GRAVITATIONAL_CONST * mass * kernel


@jit(nopython=True)
def gravity_n_spherical(
    longitude_p,
    cosphi_p,
    sinphi_p,
    radius_p,
    longitude_q,
    cosphi_q,
    sinphi_q,
    radius_q,
    mass,
):
    """
    Northing component of the gravitational acceleration due to a point source

    Returns the upward component of the gravitational acceleration produced by
    a single point source on a single computation point

    Use spherical coordinates
    """
    distance, _, coslambda = distance_spherical_core(
        longitude_p,
        cosphi_p,
        sinphi_p,
        radius_p,
        longitude_q,
        cosphi_q,
        sinphi_q,
        radius_q,
    )
    delta_n = radius_q * (cosphi_p * sinphi_q - sinphi_p * cosphi_q * coslambda)
    kernel = delta_n / distance**3
    return GRAVITATIONAL_CONST * mass * kernel


#  Tensor components
#  -------------------


@jit(nopython=True)
def gravity_ee_spherical(
    longitude_p,
    cosphi_p,
    sinphi_p,
    radius_p,
    longitude_q,
    cosphi_q,
    sinphi_q,
    radius_q,
    mass,
):
    """
    Easting-easting component of the gravitational acceleration due to a point
    source

    Returns the upward component of the gravitational acceleration produced by
    a single point source on a single computation point

    Use spherical coordinates
    """
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
    delta_e = radius_q * cosphi_q * np.sin(longitude_q - longitude_p)
    kernel = -1 / distance**3 + 3 * delta_e**2 / distance**5
    return GRAVITATIONAL_CONST * mass * kernel


@jit(nopython=True)
def gravity_nn_spherical(
    longitude_p,
    cosphi_p,
    sinphi_p,
    radius_p,
    longitude_q,
    cosphi_q,
    sinphi_q,
    radius_q,
    mass,
):
    """
    Northing-northing component of the gravitational acceleration due to a
    point source

    Returns the upward component of the gravitational acceleration produced by
    a single point source on a single computation point

    Use spherical coordinates
    """
    distance, _, coslambda = distance_spherical_core(
        longitude_p,
        cosphi_p,
        sinphi_p,
        radius_p,
        longitude_q,
        cosphi_q,
        sinphi_q,
        radius_q,
    )
    delta_n = radius_q * (cosphi_p * sinphi_q - sinphi_p * cosphi_q * coslambda)
    kernel = -1 / distance**3 + 3 * delta_n**2 / distance**5
    return GRAVITATIONAL_CONST * mass * kernel


@jit(nopython=True)
def gravity_uu_spherical(
    longitude_p,
    cosphi_p,
    sinphi_p,
    radius_p,
    longitude_q,
    cosphi_q,
    sinphi_q,
    radius_q,
    mass,
):
    """
    Upward-upward component of the gravitational acceleration due to a point
    source

    Returns the upward component of the gravitational acceleration produced by
    a single point source on a single computation point

    Use spherical coordinates
    """
    distance, cospsi, _ = distance_spherical_core(
        longitude_p,
        cosphi_p,
        sinphi_p,
        radius_p,
        longitude_q,
        cosphi_q,
        sinphi_q,
        radius_q,
    )
    delta_z = radius_p - radius_q * cospsi
    kernel = -1 / distance**3 + 3 * delta_z**2 / distance**5
    return GRAVITATIONAL_CONST * mass * kernel


@jit(nopython=True)
def gravity_en_spherical(
    longitude_p,
    cosphi_p,
    sinphi_p,
    radius_p,
    longitude_q,
    cosphi_q,
    sinphi_q,
    radius_q,
    mass,
):
    """
    Easting-northing component of the gravitational acceleration due to a point
    source

    Returns the upward component of the gravitational acceleration produced by
    a single point source on a single computation point

    Use spherical coordinates
    """
    distance, _, coslambda = distance_spherical_core(
        longitude_p,
        cosphi_p,
        sinphi_p,
        radius_p,
        longitude_q,
        cosphi_q,
        sinphi_q,
        radius_q,
    )
    delta_e = radius_q * cosphi_q * np.sin(longitude_q - longitude_p)
    delta_n = radius_q * (cosphi_p * sinphi_q - sinphi_p * cosphi_q * coslambda)
    kernel = 3 * delta_e * delta_n / distance**5
    return GRAVITATIONAL_CONST * mass * kernel


@jit(nopython=True)
def gravity_eu_spherical(
    longitude_p,
    cosphi_p,
    sinphi_p,
    radius_p,
    longitude_q,
    cosphi_q,
    sinphi_q,
    radius_q,
    mass,
):
    """
    Easting-upward component of the gravitational acceleration due to a point
    source

    Returns the upward component of the gravitational acceleration produced by
    a single point source on a single computation point

    Use spherical coordinates
    """
    distance, cospsi, _ = distance_spherical_core(
        longitude_p,
        cosphi_p,
        sinphi_p,
        radius_p,
        longitude_q,
        cosphi_q,
        sinphi_q,
        radius_q,
    )
    delta_e = radius_q * cosphi_q * np.sin(longitude_q - longitude_p)
    delta_z = radius_p - radius_q * cospsi
    kernel = 3 * delta_e * delta_z / distance**5
    return GRAVITATIONAL_CONST * mass * kernel


@jit(nopython=True)
def gravity_nu_spherical(
    longitude_p,
    cosphi_p,
    sinphi_p,
    radius_p,
    longitude_q,
    cosphi_q,
    sinphi_q,
    radius_q,
    mass,
):
    """
    Northing-upward component of the gravitational acceleration due to a point
    source

    Returns the upward component of the gravitational acceleration produced by
    a single point source on a single computation point

    Use spherical coordinates
    """
    distance, cospsi, coslambda = distance_spherical_core(
        longitude_p,
        cosphi_p,
        sinphi_p,
        radius_p,
        longitude_q,
        cosphi_q,
        sinphi_q,
        radius_q,
    )
    delta_n = radius_q * (cosphi_p * sinphi_q - sinphi_p * cosphi_q * coslambda)
    delta_z = radius_p - radius_q * cospsi
    kernel = 3 * delta_n * delta_z / distance**5
    return GRAVITATIONAL_CONST * mass * kernel
