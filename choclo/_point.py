# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Kernel functions for point sources
"""
from numba import jit

from ._distance import distance_cartesian


@jit(nopython=True)
def kernel_point_potential(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q
):
    r"""
    Kernel for the potential field due to a point source

    .. important ::

        The observation point and the location of the point source must be in
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
        Easting coordinate of the point source location.
    northing_q : float
        Northing coordinate of the point source location.
    upward_q : float
        Upward coordinate of the point source location.

    Returns
    -------
    kernel : float
        Value of the kernel function for the potential field of a point source.

    Notes
    -----
    Given an observation point located in :math:`\mathbf{p} = (x_p, y_p, z_p)`
    and a point source located in :math:`\mathbf{q} = (x_q, y_q, z_q)` defined
    in a Cartesian coordinate system, compute the kernel function for the
    potential field that the point source generates on the observation point:

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
    Kernel for easting component of the gradient due to a point source

    .. important ::

        The observation point and the location of the point source must be in
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
        Easting coordinate of the point source location.
    northing_q : float
        Northing coordinate of the point source location.
    upward_q : float
        Upward coordinate of the point source location.

    Returns
    -------
    kernel : float
        Value of the kernel function for the easting component of the
        gradient of the potential field of a point source.

    Notes
    -----
    Given an observation point located in :math:`\mathbf{p} = (x_p, y_p, z_p)`
    and a point source located in :math:`\mathbf{q} = (x_q, y_q, z_q)` defined
    in a Cartesian coordinate system, compute the kernel function for the
    easting component of the gradient of the potential field that the point
    source generates on the observation point:

    .. math::

        k_{g_\text{upward}}(\mathbf{p}, \mathbf{q}) =
        - \frac{
            x_p - x_q
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
    Kernel for northing component of the gradient due to a point source

    .. important ::

        The observation point and the location of the point source must be in
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
        Easting coordinate of the point source location.
    northing_q : float
        Northing coordinate of the point source location.
    upward_q : float
        Upward coordinate of the point source location.

    Returns
    -------
    kernel : float
        Value of the kernel function for the northing component of the
        gradient of the potential field of a point source.

    Notes
    -----
    Given an observation point located in :math:`\mathbf{p} = (x_p, y_p, z_p)`
    and a point source located in :math:`\mathbf{q} = (x_q, y_q, z_q)` defined
    in a Cartesian coordinate system, compute the kernel function for the
    northing component of the gradient of the potential field that the point
    source generates on the observation point:

    .. math::

        k_{g_\text{upward}}(\mathbf{p}, \mathbf{q}) =
        - \frac{
            y_p - y_q
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
    Kernel for upward component of the gradient due to a point source

    .. important ::

        The observation point and the location of the point source must be in
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
        Easting coordinate of the point source location.
    northing_q : float
        Northing coordinate of the point source location.
    upward_q : float
        Upward coordinate of the point source location.

    Returns
    -------
    kernel : float
        Value of the kernel function for the upward component of the gradient
        of the potential field of a point source.

    Notes
    -----
    Given an observation point located in :math:`\mathbf{p} = (x_p, y_p, z_p)`
    and a point source located in :math:`\mathbf{q} = (x_q, y_q, z_q)` defined
    in a Cartesian coordinate system, compute the kernel function for the
    upward component of the gradient of the potential field that the point
    source generates on the observation point:

    .. math::

        k_{g_\text{upward}}(\mathbf{p}, \mathbf{q}) =
        - \frac{
            z_p - z_q
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


@jit(nopython=True)
def kernel_point_g_ee(easting_p, northing_p, upward_p, easting_q, northing_q, upward_q):
    r"""
    Kernel for the :math:`G_\text{ee}` tensor component due to a point source

    .. important ::

        The observation point and the location of the point source must be in
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
        Easting coordinate of the point source location.
    northing_q : float
        Northing coordinate of the point source location.
    upward_q : float
        Upward coordinate of the point source location.

    Returns
    -------
    kernel : float
        Value of the kernel function for the ``G_ee`` component of the
        potential field tensor due to a point source.

    Notes
    -----
    Given an observation point located in :math:`\mathbf{p} = (x_p, y_p, z_p)`
    and a point source located in :math:`\mathbf{q} = (x_q, y_q, z_q)` defined
    in a Cartesian coordinate system, compute the kernel function for the
    diagonal :math:`G_\text{ee}` component of the potential field tensor that
    the point source generates on the observation point:

    .. math::

        k_{G_\text{ee}}(\mathbf{p}, \mathbf{q}) =
        \frac{
            3 (x_p - x_q)^2
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }
        - \frac{
            1
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^3
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`) and
    :math:`G_\text{ee}` is defined as:

    .. math::

        G_\text{ee}(\mathbf{p}) = \frac{\partial^2 V(\mathbf{p})}{\partial x^2}
    """
    distance = distance_cartesian(
        (easting_p, northing_p, upward_p), (easting_q, northing_q, upward_q)
    )
    return 3 * (easting_p - easting_q) ** 2 / distance**5 - 1 / distance**3


@jit(nopython=True)
def kernel_point_g_nn(easting_p, northing_p, upward_p, easting_q, northing_q, upward_q):
    r"""
    Kernel for the :math:`G_\text{nn}` tensor component due to a point source

    .. important ::

        The observation point and the location of the point source must be in
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
        Easting coordinate of the point source location.
    northing_q : float
        Northing coordinate of the point source location.
    upward_q : float
        Upward coordinate of the point source location.

    Returns
    -------
    kernel : float
        Value of the kernel function for the ``G_nn`` component of the
        potential field tensor due to a point source.

    Notes
    -----
    Given an observation point located in :math:`\mathbf{p} = (x_p, y_p, z_p)`
    and a point source located in :math:`\mathbf{q} = (x_q, y_q, z_q)` defined
    in a Cartesian coordinate system, compute the kernel function for the
    diagonal :math:`G_\text{nn}` component of the potential field tensor that
    the point source generates on the observation point:

    .. math::

        k_{G_\text{nn}}(\mathbf{p}, \mathbf{q}) =
        \frac{
            3 (y_p - y_q)^2
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }
        - \frac{
            1
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^3
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`) and
    :math:`G_\text{nn}` is defined as:

    .. math::

        G_\text{ee}(\mathbf{p}) = \frac{\partial^2 V(\mathbf{p})}{\partial x^2}
    """
    distance = distance_cartesian(
        (easting_p, northing_p, upward_p), (easting_q, northing_q, upward_q)
    )
    return 3 * (northing_p - northing_q) ** 2 / distance**5 - 1 / distance**3


@jit(nopython=True)
def kernel_point_g_zz(easting_p, northing_p, upward_p, easting_q, northing_q, upward_q):
    r"""
    Kernel for the :math:`G_\text{zz}` tensor component due to a point source

    .. important ::

        The observation point and the location of the point source must be in
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
        Easting coordinate of the point source location.
    northing_q : float
        Northing coordinate of the point source location.
    upward_q : float
        Upward coordinate of the point source location.

    Returns
    -------
    kernel : float
        Value of the kernel function for the ``G_zz`` component of the
        potential field tensor due to a point source.

    Notes
    -----
    Given an observation point located in :math:`\mathbf{p} = (x_p, y_p, z_p)`
    and a point source located in :math:`\mathbf{q} = (x_q, y_q, z_q)` defined
    in a Cartesian coordinate system, compute the kernel function for the
    diagonal :math:`G_\text{zz}` component of the potential field tensor that
    the point source generates on the observation point:

    .. math::

        k_{G_\text{zz}}(\mathbf{p}, \mathbf{q}) =
        \frac{
            3 (z_p - z_q)^2
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }
        - \frac{
            1
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^3
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`) and
    :math:`G_\text{zz}` is defined as:

    .. math::

        G_\text{zz}(\mathbf{p}) = \frac{\partial^2 V(\mathbf{p})}{\partial z^2}
    """
    distance = distance_cartesian(
        (easting_p, northing_p, upward_p), (easting_q, northing_q, upward_q)
    )
    return 3 * (upward_p - upward_q) ** 2 / distance**5 - 1 / distance**3


@jit(nopython=True)
def kernel_point_g_en(easting_p, northing_p, upward_p, easting_q, northing_q, upward_q):
    r"""
    Kernel for the :math:`G_\text{en}` tensor component due to a point source

    .. important ::

        The observation point and the location of the point source must be in
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
        Easting coordinate of the point source location.
    northing_q : float
        Northing coordinate of the point source location.
    upward_q : float
        Upward coordinate of the point source location.

    Returns
    -------
    kernel : float
        Value of the kernel function for the ``G_en`` component of the
        potential field tensor due to a point source.

    Notes
    -----
    Given an observation point located in :math:`\mathbf{p} = (x_p, y_p, z_p)`
    and a point source located in :math:`\mathbf{q} = (x_q, y_q, z_q)` defined
    in a Cartesian coordinate system, compute the kernel function for the
    non-diagonal :math:`G_\text{en}` component of the potential field tensor
    that the point source generates on the observation point:

    .. math::

        k_{G_\text{en}}(\mathbf{p}, \mathbf{q}) =
        \frac{
            3 (x_p - x_q) * (y_p - y_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`) and
    :math:`G_\text{en}` is defined as:

    .. math::

        G_\text{en}(\mathbf{p}) =
        \frac{
            \partial^2 V(\mathbf{p})
        }{
            \partial x \partial y
        }
    """
    distance = distance_cartesian(
        (easting_p, northing_p, upward_p), (easting_q, northing_q, upward_q)
    )
    return 3 * (easting_p - easting_q) * (northing_p - northing_q) / distance**5


@jit(nopython=True)
def kernel_point_g_ez(easting_p, northing_p, upward_p, easting_q, northing_q, upward_q):
    r"""
    Kernel for the :math:`G_\text{ez}` tensor component due to a point source

    .. important ::

        The observation point and the location of the point source must be in
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
        Easting coordinate of the point source location.
    northing_q : float
        Northing coordinate of the point source location.
    upward_q : float
        Upward coordinate of the point source location.

    Returns
    -------
    kernel : float
        Value of the kernel function for the ``G_ez`` component of the
        potential field tensor due to a point source.

    Notes
    -----
    Given an observation point located in :math:`\mathbf{p} = (x_p, y_p, z_p)`
    and a point source located in :math:`\mathbf{q} = (x_q, y_q, z_q)` defined
    in a Cartesian coordinate system, compute the kernel function for the
    non-diagonal :math:`G_\text{ez}` component of the potential field tensor
    that the point source generates on the observation point:

    .. math::

        k_{G_\text{ez}}(\mathbf{p}, \mathbf{q}) =
        \frac{
            3 (x_p - x_q) * (z_p - z_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`) and
    :math:`G_\text{en}` is defined as:

    .. math::

        G_\text{ez}(\mathbf{p}) =
        \frac{
            \partial^2 V(\mathbf{p})
        }{
            \partial x \partial z
        }
    """
    distance = distance_cartesian(
        (easting_p, northing_p, upward_p), (easting_q, northing_q, upward_q)
    )
    return 3 * (easting_p - easting_q) * (upward_p - upward_q) / distance**5


@jit(nopython=True)
def kernel_point_g_nz(easting_p, northing_p, upward_p, easting_q, northing_q, upward_q):
    r"""
    Kernel for the :math:`G_\text{ez}` tensor component due to a point source

    .. important ::

        The observation point and the location of the point source must be in
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
        Easting coordinate of the point source location.
    northing_q : float
        Northing coordinate of the point source location.
    upward_q : float
        Upward coordinate of the point source location.

    Returns
    -------
    kernel : float
        Value of the kernel function for the ``G_nz`` component of the
        potential field tensor due to a point source.

    Notes
    -----
    Given an observation point located in :math:`\mathbf{p} = (x_p, y_p, z_p)`
    and a point source located in :math:`\mathbf{q} = (x_q, y_q, z_q)` defined
    in a Cartesian coordinate system, compute the kernel function for the
    non-diagonal :math:`G_\text{nz}` component of the potential field tensor
    that the point source generates on the observation point:

    .. math::

        k_{G_\text{nz}}(\mathbf{p}, \mathbf{q}) =
        \frac{
            3 (y_p - y_q) * (z_p - z_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`) and
    :math:`G_\text{en}` is defined as:

    .. math::

        G_\text{nz}(\mathbf{p}) =
        \frac{
            \partial^2 V(\mathbf{p})
        }{
            \partial y \partial z
        }
    """
    distance = distance_cartesian(
        (easting_p, northing_p, upward_p), (easting_q, northing_q, upward_q)
    )
    return 3 * (northing_p - northing_q) * (upward_p - upward_q) / distance**5