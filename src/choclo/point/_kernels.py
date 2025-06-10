# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Kernel functions for point sources.
"""

from numba import jit


@jit(nopython=True)
def kernel_pot(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Compute the inverse of the distance between the two points.

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
    Easting component of the gradient of the inverse of the distance.

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
    Northing component of the gradient of the inverse of the distance.

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
    Upward component of the gradient of the inverse of the distance.

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
    Second derivative of the inverse of the distance along easting-easting.

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
    Second derivative of the inverse of the distance along northing-northing.

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
    Second derivative of the inverse of the distance along upward-upward.

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
    Second derivative of the inverse of the distance along easting-northing.

    This is equivalent to the derivative along northing-easting.

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
        k_{yx}(\mathbf{p}, \mathbf{q}) =
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
    Second derivative of the inverse of the distance along easting-upward.

    This is equivalent to the derivative along upward-easting.

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
        k_{zx}(\mathbf{p}, \mathbf{q}) =
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
    Second derivative of the inverse of the distance along northing-upward.

    This is equivalent to the derivative along upward-northing.
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
        k_{zy}(\mathbf{p}, \mathbf{q}) =
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


@jit(nopython=True)
def kernel_eee(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Third derivative of the inverse of the distance along east-east-east.

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

        k_{xxx}(\mathbf{p}, \mathbf{q}) =
        \frac{\partial^3}{\partial x_p^3}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
        \frac{
            9 (x_p - x_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }
        - \frac{
            15 (x_p - x_q)^3
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^7
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    easting = easting_p - easting_q
    return 9 * easting / distance**5 - 15 * easting**3 / distance**7


@jit(nopython=True)
def kernel_nnn(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Third derivative of the inverse of the distance along north-north-north.

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

        k_{yyy}(\mathbf{p}, \mathbf{q}) =
        \frac{\partial^3}{\partial y_p^3}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
        \frac{
            9 (y_p - y_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }
        - \frac{
            15 (y_p - y_q)^3
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^7
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    northing = northing_p - northing_q
    return 9 * northing / distance**5 - 15 * northing**3 / distance**7


@jit(nopython=True)
def kernel_uuu(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Third derivative of the inverse of the distance along up-up-up.

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

        k_{zzz}(\mathbf{p}, \mathbf{q}) =
        \frac{\partial^3}{\partial z_p^3}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
        \frac{
            9 (z_p - z_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }
        - \frac{
            15 (z_p - z_q)^3
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^7
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    upward = upward_p - upward_q
    return 9 * upward / distance**5 - 15 * upward**3 / distance**7


@jit(nopython=True)
def kernel_een(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Third derivative of the inverse of the distance along east-east-north.

    This is equivalent to the derivatives along east-north-east and
    north-east-east.

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

        k_{xxy}(\mathbf{p}, \mathbf{q}) =
        k_{xyx} =
        k_{yxx} =
        \frac{\partial^3}{\partial x_p^2 \partial y_p}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
        \frac{
            3 (y_p - y_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }
        - \frac{
            15 (x_p - x_q)^2 (y_p - y_q)

        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^7
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    easting = easting_p - easting_q
    northing = northing_p - northing_q
    return 3 * northing / distance**5 - 15 * easting**2 * northing / distance**7


@jit(nopython=True)
def kernel_eeu(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Third derivative of the inverse of the distance along east-east-up.

    This is equivalent to the derivatives along east-up-east and up-east-east.

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

        k_{xxz}(\mathbf{p}, \mathbf{q}) =
        k_{xzx} =
        k_{zxx} =
        \frac{\partial^3}{\partial x_p^2 \partial z_p}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
        \frac{
            3 (z_p - z_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }
        - \frac{
            15 (x_p - x_q)^2 (z_p - z_q)

        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^7
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    easting = easting_p - easting_q
    upward = upward_p - upward_q
    return 3 * upward / distance**5 - 15 * easting**2 * upward / distance**7


@jit(nopython=True)
def kernel_nne(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Third derivative of the inverse of the distance along north-north-east.

    This is equivalent to the derivatives along north-east-north and
    east-north-north.

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

        k_{yyx}(\mathbf{p}, \mathbf{q}) =
        k_{yxy} =
        k_{xyy} =
        \frac{\partial^3}{\partial y_p^2 \partial x_p}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
        \frac{
            3 (x_p - x_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }
        - \frac{
            15 (y_p - y_q)^2 (x_p - x_q)

        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^7
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    northing = northing_p - northing_q
    easting = easting_p - easting_q
    return 3 * easting / distance**5 - 15 * northing**2 * easting / distance**7


@jit(nopython=True)
def kernel_nnu(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Third derivative of the inverse of the distance along north-north-up.

    This is equivalent to the derivatives along north-up-north and
    up-north-north.

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

        k_{yyz}(\mathbf{p}, \mathbf{q}) =
        k_{yzy} =
        k_{zyy} =
        \frac{\partial^3}{\partial y_p^2 \partial x_p}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
        \frac{
            3 (z_p - z_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }
        - \frac{
            15 (y_p - y_q)^2 (z_p - z_q)

        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^7
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    northing = northing_p - northing_q
    upward = upward_p - upward_q
    return 3 * upward / distance**5 - 15 * northing**2 * upward / distance**7


@jit(nopython=True)
def kernel_uue(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Third derivative of the inverse of the distance along up-up-east.

    This is equivalent to the derivatives along up-east-up and east-up-up.

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

        k_{zzx}(\mathbf{p}, \mathbf{q}) =
        k_{zxz} =
        k_{xzz} =
        \frac{\partial^3}{\partial z_p^2 \partial x_p}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
        \frac{
            3 (x_p - x_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }
        - \frac{
            15 (z_p - z_q)^2 (x_p - x_q)

        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^7
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    upward = upward_p - upward_q
    easting = easting_p - easting_q
    return 3 * easting / distance**5 - 15 * upward**2 * easting / distance**7


@jit(nopython=True)
def kernel_uun(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Third derivative of the inverse of the distance along up-up-north.

    This is equivalent to the derivatives along up-north-up and north-up-up.

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

        k_{zzy}(\mathbf{p}, \mathbf{q}) =
        k_{zyz} =
        k_{yzz} =
        \frac{\partial^3}{\partial z_p^2 \partial y_p}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
        \frac{
            3 (y_p - y_q)
        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^5
        }
        - \frac{
            15 (z_p - z_q)^2 (y_p - y_q)

        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^7
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    upward = upward_p - upward_q
    northing = northing_p - northing_q
    return 3 * northing / distance**5 - 15 * upward**2 * northing / distance**7


@jit(nopython=True)
def kernel_enu(
    easting_p, northing_p, upward_p, easting_q, northing_q, upward_q, distance
):
    r"""
    Third derivative of the inverse of the distance along east-north-up.

    This is equivalent to the derivatives along east-up-north, up-east-north,
    north-east-up, north-up-east, and up-north-east.

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

        k_{xyz}(\mathbf{p}, \mathbf{q}) =
        k_{xzy} =
        k_{zxy} =
        k_{yxz} =
        k_{yzx} =
        k_{zyx} =
        \frac{\partial^3}{\partial x_p \partial y_p \partial z_p}
        \left(
            \frac{1}{\lVert \mathbf{p} - \mathbf{q} \rVert_2}
        \right)
        =
        - \frac{
            15(x_p - x_q)(y_p - y_q)(z_p - z_q)

        }{
            \lVert \mathbf{p} - \mathbf{q} \rVert_2^7
        }

    where :math:`\lVert \cdot \rVert_2` refer to the :math:`L_2` norm (the
    Euclidean distance between :math:`\mathbf{p}` and :math:`\mathbf{q}`).
    """
    easting = easting_p - easting_q
    northing = northing_p - northing_q
    upward = upward_p - upward_q
    return -15 * easting * northing * upward / distance**7
