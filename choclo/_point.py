# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Kernel functions for point mass gravity forward modelling
"""
from numba import jit

from ._distance import distance_cartesian


@jit(nopython=True)
def kernel_point_potential(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q
):
    r"""
    Kernel for the potential field of a point mass

    .. important ::

        The observation point and the location of the point mass must be in
        Cartesian coordinates and have the same units.

    Parameters
    ----------
    easting_p : float
        Easting coordinate of the observation point.
    northing_p : float
        Northing coordinate of the observation point.
    upward_p : float
        Upward coordinate of the observation point.
    easting_q : float
        Easting coordinate of the point mass location.
    northing_q : float
        Northing coordinate of the point mass location.
    upward_q : float
        Upward coordinate of the point mass location.

    Returns
    -------
    kernel : float
        Value of the kernel function for the potential field of a point mass.

    Notes
    -----
    Given an observation point located in :math:`\mathbf{p} = (x_p, y_p, z_p)`
    and a point mass located in :math:`\mathbf{q} = (x_q, y_q, z_q)` defined in
    a Cartesian coordinate system, compute the kernel function for the
    potential field that the point mass generates on the observation point:

    .. math::

        k_V(\mathbf{p}, \mathbf{q}) =
        \frac{
            1
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    distance = distance_cartesian(
        (easting_p, northing_p, upward_p), (easting_q, northing_q, upward_q)
    )
    return 1 / distance


@jit(nopython=True)
def kernel_point_g_easting(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q
):
    r"""
    Kernel for easting component of the gradient of the field of a point mass

    .. important ::

        The observation point and the location of the point mass must be in
        Cartesian coordinates and have the same units.

    Parameters
    ----------
    easting_p : float
        Easting coordinate of the observation point.
    northing_p : float
        Northing coordinate of the observation point.
    upward_p : float
        Upward coordinate of the observation point.
    easting_q : float
        Easting coordinate of the point mass location.
    northing_q : float
        Northing coordinate of the point mass location.
    upward_q : float
        Upward coordinate of the point mass location.

    Returns
    -------
    kernel : float
        Value of the kernel function for the easting component of the
        gradient of the potential field of a point mass.

    Notes
    -----
    Given an observation point located in :math:`\mathbf{p} = (x_p, y_p, z_p)`
    and a point mass located in :math:`\mathbf{q} = (x_q, y_q, z_q)` defined in
    a Cartesian coordinate system, compute the kernel function for the easting
    component of the gradient of the potential field that the point mass
    generates on the observation point:

    .. math::

        k_{g_\text{upward}}(\mathbf{p}, \mathbf{q}) =
        \frac{
            - (x_p - x_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    distance = distance_cartesian(
        (easting_p, northing_p, upward_p), (easting_q, northing_q, upward_q)
    )
    return -(easting_p - easting_q) / distance**3


@jit(nopython=True)
def kernel_point_g_northing(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q
):
    r"""
    Kernel for northing component of the gradient of the field of a point mass

    .. important ::

        The observation point and the location of the point mass must be in
        Cartesian coordinates

    Parameters
    ----------
    easting_p : float
        Easting coordinate of the observation point.
    northing_p : float
        Northing coordinate of the observation point.
    upward_p : float
        Upward coordinate of the observation point.
    easting_q : float
        Easting coordinate of the point mass location.
    northing_q : float
        Northing coordinate of the point mass location.
    upward_q : float
        Upward coordinate of the point mass location.

    Returns
    -------
    kernel : float
        Value of the kernel function for the northing component of the
        gradient of the potential field of a point mass.

    Notes
    -----
    Given an observation point located in :math:`\mathbf{p} = (x_p, y_p, z_p)`
    and a point mass located in :math:`\mathbf{q} = (x_q, y_q, z_q)` defined in
    a Cartesian coordinate system, compute the kernel function for the northing
    component of the gradient of the potential field that the point mass
    generates on the observation point:

    .. math::

        k_{g_\text{upward}}(\mathbf{p}, \mathbf{q}) =
        \frac{
            - (y_p - y_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    distance = distance_cartesian(
        (easting_p, northing_p, upward_p), (easting_q, northing_q, upward_q)
    )
    return -(northing_p - northing_q) / distance**3


@jit(nopython=True)
def kernel_point_g_upward(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q
):
    r"""
    Kernel for upward component of the gradient of the field of a point mass

    .. important ::

        The observation point and the location of the point mass must be in
        Cartesian coordinates and have the same units.

    .. warning ::

        This kernel corresponds to the **upward** component of the
        gradient of the potential field. In need of the _downward_ component,
        multiply this result by -1.

    Parameters
    ----------
    easting_p : float
        Easting coordinate of the observation point.
    northing_p : float
        Northing coordinate of the observation point.
    upward_p : float
        Upward coordinate of the observation point.
    easting_q : float
        Easting coordinate of the point mass location.
    northing_q : float
        Northing coordinate of the point mass location.
    upward_q : float
        Upward coordinate of the point mass location.

    Returns
    -------
    kernel : float
        Value of the kernel function for the upward component of the gradient
        of the potential field of a point mass.

    Notes
    -----
    Given an observation point located in :math:`\mathbf{p} = (x_p, y_p, z_p)`
    and a point mass located in :math:`\mathbf{q} = (x_q, y_q, z_q)` defined in
    a Cartesian coordinate system, compute the kernel function for the upward
    component of the gradient of the potential field that the point mass
    generates on the observation point:

    .. math::

        k_{g_\text{upward}}(\mathbf{p}, \mathbf{q}) =
        \frac{
            - (z_p - z_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    distance = distance_cartesian(
        (easting_p, northing_p, upward_p), (easting_q, northing_q, upward_q)
    )
    return -(upward_p - upward_q) / distance**3
