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


@jit(nopython=True)
def kernel_pot(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    The inverse of the distance between the two points

    .. important ::

        The coordinates of the two points must be in Cartesian coordinates and
        have the same units.

    Parameters
    ----------
    easting_p, northing_p, upward_p : float
        Easting, northing and upward coordinates of point :math:`\mathbf{p}`.
    easting_q, northing_q, upward_q : float
        Easting, northing and upward coordinates of point :math:`\mathbf{q}`.
    distance : float
        Euclidean distance between points :math:`\mathbf{p}` and
        :math:`\mathbf{q}`.

    Returns
    -------
    kernel : float
        Value of the kernel function.

    Notes
    -----
    Given two points :math:`\mathbf{p} = (x_p, y_p, z_p)` and :math:`\mathbf{q}
    = (x_q, y_q, z_q)` defined in a Cartesian coordinate system, compute the
    following kernel function:

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
    return 1 / distance


@jit(nopython=True)
def kernel_e(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Easting component of the gradient of the inverse of the distance

    .. important ::

        The coordinates of the two points must be in Cartesian coordinates and
        have the same units.

    Parameters
    ----------
    easting_p, northing_p, upward_p : float
        Easting, northing and upward coordinates of point :math:`\mathbf{p}`.
    easting_q, northing_q, upward_q : float
        Easting, northing and upward coordinates of point :math:`\mathbf{q}`.
    distance : float
        Euclidean distance between points :math:`\mathbf{p}` and
        :math:`\mathbf{q}`.

    Returns
    -------
    kernel : float
        Value of the kernel function.

    Notes
    -----
    Given two points :math:`\mathbf{p} = (x_p, y_p, z_p)` and :math:`\mathbf{q}
    = (x_q, y_q, z_q)` defined in a Cartesian coordinate system, compute the
    following kernel function:

    .. math::

        k_x(\mathbf{p}, \mathbf{q}) =
        \frac{\partial}{\partial x}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
        - \frac{
            x_p - x_q
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    return -(easting_p - easting_q) / distance**3


@jit(nopython=True)
def kernel_n(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Northing component of the gradient of the inverse of the distance

    .. important ::

        The coordinates of the two points must be in Cartesian coordinates and
        have the same units.

    Parameters
    ----------
    easting_p, northing_p, upward_p : float
        Easting, northing and upward coordinates of point :math:`\mathbf{p}`.
    easting_q, northing_q, upward_q : float
        Easting, northing and upward coordinates of point :math:`\mathbf{q}`.
    distance : float
        Euclidean distance between points :math:`\mathbf{p}` and
        :math:`\mathbf{q}`.

    Returns
    -------
    kernel : float
        Value of the kernel function.

    Notes
    -----
    Given two points :math:`\mathbf{p} = (x_p, y_p, z_p)` and :math:`\mathbf{q}
    = (x_q, y_q, z_q)` defined in a Cartesian coordinate system, compute the
    following kernel function:

    .. math::

        k_y(\mathbf{p}, \mathbf{q}) =
        \frac{\partial}{\partial y}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
        - \frac{
            y_p - y_q
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    return -(northing_p - northing_q) / distance**3


@jit(nopython=True)
def kernel_u(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Upward component of the gradient of the inverse of the distance

    .. important ::

        The coordinates of the two points must be in Cartesian coordinates and
        have the same units.

    Parameters
    ----------
    easting_p, northing_p, upward_p : float
        Easting, northing and upward coordinates of point :math:`\mathbf{p}`.
    easting_q, northing_q, upward_q : float
        Easting, northing and upward coordinates of point :math:`\mathbf{q}`.
    distance : float
        Euclidean distance between points :math:`\mathbf{p}` and
        :math:`\mathbf{q}`.

    Returns
    -------
    kernel : float
        Value of the kernel function.

    Notes
    -----
    Given two points :math:`\mathbf{p} = (x_p, y_p, z_p)` and :math:`\mathbf{q}
    = (x_q, y_q, z_q)` defined in a Cartesian coordinate system, compute the
    following kernel function:

    .. math::

        k_z(\mathbf{p}, \mathbf{q}) =
        \frac{\partial}{\partial z}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
        - \frac{
            z_p - z_q
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    return -(upward_p - upward_q) / distance**3


@jit(nopython=True)
def kernel_ee(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Second derivative of the inverse of the distance along easting-easting

    .. important ::

        The coordinates of the two points must be in Cartesian coordinates and
        have the same units.

    Parameters
    ----------
    easting_p, northing_p, upward_p : float
        Easting, northing and upward coordinates of point :math:`\mathbf{p}`.
    easting_q, northing_q, upward_q : float
        Easting, northing and upward coordinates of point :math:`\mathbf{q}`.
    distance : float
        Euclidean distance between points :math:`\mathbf{p}` and
        :math:`\mathbf{q}`.

    Returns
    -------
    kernel : float
        Value of the kernel function.

    Notes
    -----
    Given two points :math:`\mathbf{p} = (x_p, y_p, z_p)` and :math:`\mathbf{q}
    = (x_q, y_q, z_q)` defined in a Cartesian coordinate system, compute the
    following kernel function:

    .. math::

        k_{xx}(\mathbf{p}, \mathbf{q}) =
        \frac{\partial^2}{\partial x^2}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
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
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    return 3 * (easting_p - easting_q) ** 2 / distance**5 - 1 / distance**3


@jit(nopython=True)
def kernel_nn(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Second derivative of the inverse of the distance along northing-northing

    .. important ::

        The coordinates of the two points must be in Cartesian coordinates and
        have the same units.

    Parameters
    ----------
    easting_p, northing_p, upward_p : float
        Easting, northing and upward coordinates of point :math:`\mathbf{p}`.
    easting_q, northing_q, upward_q : float
        Easting, northing and upward coordinates of point :math:`\mathbf{q}`.
    distance : float
        Euclidean distance between points :math:`\mathbf{p}` and
        :math:`\mathbf{q}`.

    Returns
    -------
    kernel : float
        Value of the kernel function.

    Notes
    -----
    Given two points :math:`\mathbf{p} = (x_p, y_p, z_p)` and :math:`\mathbf{q}
    = (x_q, y_q, z_q)` defined in a Cartesian coordinate system, compute the
    following kernel function:

    .. math::

        k_{yy}(\mathbf{p}, \mathbf{q}) =
        \frac{\partial^2}{\partial y^2}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
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
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    return 3 * (northing_p - northing_q) ** 2 / distance**5 - 1 / distance**3


@jit(nopython=True)
def kernel_uu(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Second derivative of the inverse of the distance along upward-upward

    .. important ::

        The coordinates of the two points must be in Cartesian coordinates and
        have the same units.

    Parameters
    ----------
    easting_p, northing_p, upward_p : float
        Easting, northing and upward coordinates of point :math:`\mathbf{p}`.
    easting_q, northing_q, upward_q : float
        Easting, northing and upward coordinates of point :math:`\mathbf{q}`.
    distance : float
        Euclidean distance between points :math:`\mathbf{p}` and
        :math:`\mathbf{q}`.

    Returns
    -------
    kernel : float
        Value of the kernel function.

    Notes
    -----
    Given two points :math:`\mathbf{p} = (x_p, y_p, z_p)` and :math:`\mathbf{q}
    = (x_q, y_q, z_q)` defined in a Cartesian coordinate system, compute the
    following kernel function:

    .. math::

        k_{zz}(\mathbf{p}, \mathbf{q}) =
        \frac{\partial^2}{\partial z^2}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
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
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    return 3 * (upward_p - upward_q) ** 2 / distance**5 - 1 / distance**3


@jit(nopython=True)
def kernel_en(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Second derivative of the inverse of the distance along easting-northing

    .. important ::

        The coordinates of the two points must be in Cartesian coordinates and
        have the same units.

    Parameters
    ----------
    easting_p, northing_p, upward_p : float
        Easting, northing and upward coordinates of point :math:`\mathbf{p}`.
    easting_q, northing_q, upward_q : float
        Easting, northing and upward coordinates of point :math:`\mathbf{q}`.
    distance : float
        Euclidean distance between points :math:`\mathbf{p}` and
        :math:`\mathbf{q}`.

    Returns
    -------
    kernel : float
        Value of the kernel function.

    Notes
    -----
    Given two points :math:`\mathbf{p} = (x_p, y_p, z_p)` and :math:`\mathbf{q}
    = (x_q, y_q, z_q)` defined in a Cartesian coordinate system, compute the
    following kernel function:

    .. math::

        k_{xy}(\mathbf{p}, \mathbf{q}) =
        \frac{\partial}{\partial x \partial y}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
        \frac{
            3 (x_p - x_q) (y_p - y_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    return 3 * (easting_p - easting_q) * (northing_p - northing_q) / distance**5


@jit(nopython=True)
def kernel_eu(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Second derivative of the inverse of the distance along easting-upward

    .. important ::

        The coordinates of the two points must be in Cartesian coordinates and
        have the same units.

    Parameters
    ----------
    easting_p, northing_p, upward_p : float
        Easting, northing and upward coordinates of point :math:`\mathbf{p}`.
    easting_q, northing_q, upward_q : float
        Easting, northing and upward coordinates of point :math:`\mathbf{q}`.
    distance : float
        Euclidean distance between points :math:`\mathbf{p}` and
        :math:`\mathbf{q}`.

    Returns
    -------
    kernel : float
        Value of the kernel function.

    Notes
    -----
    Given two points :math:`\mathbf{p} = (x_p, y_p, z_p)` and :math:`\mathbf{q}
    = (x_q, y_q, z_q)` defined in a Cartesian coordinate system, compute the
    following kernel function:

    .. math::

        k_{xz}(\mathbf{p}, \mathbf{q}) =
        \frac{\partial}{\partial x \partial z}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
        \frac{
            3 (x_p - x_q) (z_p - z_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    return 3 * (easting_p - easting_q) * (upward_p - upward_q) / distance**5


@jit(nopython=True)
def kernel_nu(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Second derivative of the inverse of the distance along northing-upward

    .. important ::

        The coordinates of the two points must be in Cartesian coordinates and
        have the same units.

    Parameters
    ----------
    easting_p, northing_p, upward_p : float
        Easting, northing and upward coordinates of point :math:`\mathbf{p}`.
    easting_q, northing_q, upward_q : float
        Easting, northing and upward coordinates of point :math:`\mathbf{q}`.
    distance : float
        Euclidean distance between points :math:`\mathbf{p}` and
        :math:`\mathbf{q}`.

    Returns
    -------
    kernel : float
        Value of the kernel function.

    Notes
    -----
    Given two points :math:`\mathbf{p} = (x_p, y_p, z_p)` and :math:`\mathbf{q}
    = (x_q, y_q, z_q)` defined in a Cartesian coordinate system, compute the
    following kernel function:

    .. math::

        k_{yz}(\mathbf{p}, \mathbf{q}) =
        \frac{\partial}{\partial y \partial z}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
        \frac{
            3 (y_p - y_q) (z_p - z_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    return 3 * (northing_p - northing_q) * (upward_p - upward_q) / distance**5
