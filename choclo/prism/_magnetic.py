# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Magnetic forward modelling function for rectangular prisms
"""
import numpy as np
from numba import jit

from ..constants import VACUUM_MAGNETIC_PERMEABILITY
from ._kernels import (
    kernel_ee,
    kernel_en,
    kernel_eu,
    kernel_nn,
    kernel_nu,
    kernel_uu,
)


@jit(nopython=True)
def magnetic_field(easting, northing, upward, prism, magnetization):
    r"""
    Magnetic field due to a rectangular prism

    Returns the three components of the magnetic field due to a single
    rectangular prism on a single computation point.

    .. note::

        Use this function when all the three component of the magnetic fields
        are needed. Running this function is faster than computing each
        component separately. Use one of :func:`magnetic_e`,
        :func:`magnetic_n`, :func:`magnetic_u` if you need only one of them.

    Parameters
    ----------
    easting : float
        Easting coordinate of the observation point. Must be in meters.
    northing : float
        Northing coordinate of the observation point. Must be in meters.
    upward : float
        Upward coordinate of the observation point. Must be in meters.
    prism : 1d-array
        One dimensional array containing the coordinates of the prism in the
        following order: ``west``, ``east``, ``south``, ``north``, ``bottom``,
        ``top`` in a Cartesian coordinate system.
        All coordinates should be in meters.
    magnetization : 1d-array
        Magnetization vector of the prism. It should have three components in
        the following order: ``magnetization_easting``,
        ``magnetization_northing``, ``magnetization_upward``.
        Should be in :math:`A m^{-1}`.

    Returns
    -------
    b : array
        Array containing the three components of the magnetic field generated
        by the prism on the observation point in :math:`\text{T}`.
        The components are returned in the following order: ``b_e``, ``b_n``,
        ``b_u``.

    Notes
    -----

    References
    ----------
    - [Oliveira2015]_
    - [Nagy2000]_
    - [Nagy2002]_
    - [Fukushima2020]_
    """
    # Initialize magnetic field vector
    b = np.zeros(3, dtype=np.float64)
    # Precompute the volume of the prism
    volume = (prism[1] - prism[0]) * (prism[3] - prism[2]) * (prism[5] - prism[4])
    # Iterate over the vertices of the prism
    for i in range(2):
        # Compute shifted easting coordinate
        shift_east = prism[1 - i] - easting
        shift_east_sq = shift_east**2
        for j in range(2):
            # Compute shifted northing coordinate
            shift_north = prism[3 - j] - northing
            shift_north_sq = shift_north**2
            for k in range(2):
                # Compute shifted upward coordinate
                shift_upward = prism[5 - k] - upward
                shift_upward_sq = shift_upward**2
                # Compute the radius
                radius = np.sqrt(shift_east_sq + shift_north_sq + shift_upward_sq)
                # Compute all kernel tensor components for the current vertex
                k_ee = kernel_ee(shift_east, shift_north, shift_upward, radius)
                k_nn = kernel_nn(shift_east, shift_north, shift_upward, radius)
                k_uu = kernel_uu(shift_east, shift_north, shift_upward, radius)
                k_en = kernel_en(shift_east, shift_north, shift_upward, radius)
                k_eu = kernel_eu(shift_east, shift_north, shift_upward, radius)
                k_nu = kernel_nu(shift_east, shift_north, shift_upward, radius)
                # Get the sign of this terms based on the current vertex
                sign = (-1) ** (i + j + k)
                # Compute the dot product between the kernel tensor and the
                # magnetization vector of the prism
                b[0] += sign * (
                    magnetization[0] * k_ee
                    + magnetization[1] * k_en
                    + magnetization[2] * k_eu
                )
                b[1] += sign * (
                    magnetization[0] * k_en
                    + magnetization[1] * k_nn
                    + magnetization[2] * k_nu
                )
                b[2] += sign * (
                    magnetization[0] * k_eu
                    + magnetization[1] * k_nu
                    + magnetization[2] * k_uu
                )
    b *= VACUUM_MAGNETIC_PERMEABILITY / 4 / np.pi / volume
    return b