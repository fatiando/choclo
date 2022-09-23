# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Forward modelling function for rectangular prisms
"""
from numba import jit


@jit(nopython=True)
def _evaluate_kernel(easting, northing, upward, prism, kernel):
    r"""
    Evaluate a kernel function on every shifted vertex of a prism

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
    kernel : callable
        Kernel function that will be evaluated on each one of the shifted
        vertices of the prism.

    Returns
    -------
    result : float
        Evaluation of the kernel function on each one of the vertices of the
        prism.

    Notes
    -----
    This function evaluates a numerical kernel :math:`k(x, y, z)` on each one
    of the vertices of the prism:

    .. math::

        v(\mathbf{p}) =
            \lVert \lVert \lVert
            k(x, y, z)
            \lVert\limits_{x_1}^{x_2}
            \lVert\limits_{y_1}^{y_2}
            \lVert\limits_{z_1}^{z_2}

    where :math:`x_1`, :math:`x_2`, :math:`y_1`, :math:`y_2`, :math:`z_1` and
    :math:`z_2` are boundaries of the rectangular prism in the *shifted
    coordinates* defined by the Cartesian coordinate system with its origin
    located on the observation point :math:`\mathbf{p}`.

    References
    ----------
    - [Nagy2000]_
    - [Nagy2002]_
    - [Fukushima2020]_
    """
    # Initialize result float to zero
    result = 0
    # Iterate over the vertices of the prism
    for i in range(2):
        # Compute shifted easting coordinate
        shift_east = prism[1 - i] - easting
        for j in range(2):
            # Compute shifted northing coordinate
            shift_north = prism[3 - j] - northing
            for k in range(2):
                # Compute shifted upward coordinate
                shift_upward = prism[5 - k] - upward
                # If i, j or k is 1, the corresponding shifted
                # coordinate will refer to the lower boundary,
                # meaning the corresponding term should have a minus
                # sign.
                result += (-1) ** (i + j + k) * kernel(
                    shift_east, shift_north, shift_upward
                )
    return result
