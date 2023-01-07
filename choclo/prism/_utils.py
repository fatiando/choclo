# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Utility functions for forward modelling of prisms
"""
from numba import jit


@jit(nopython=True)
def is_point_on_easting_edge(easting, northing, upward, prism):
    """
    Check if observation point falls in a prism edge parallel to easting

    Return True if the observation point falls in one of the prism edges
    parallel to the easting direction or in any one of the vertices of the
    prism. Return False if otherwise.

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

    Returns
    -------
    result : bool
        Return True if the observation point falls in one of the prism edges
        parallel to the easting direction or in any one of the vertices of the
        prism. Return False if otherwise.
    """
    in_easting = prism[0] <= easting <= prism[1]
    in_northing = (northing == prism[2]) or (northing == prism[3])
    in_upward = (upward == prism[4]) or (upward == prism[5])
    if in_easting and in_northing and in_upward:
        return True
    return False


@jit(nopython=True)
def is_point_on_northing_edge(easting, northing, upward, prism):
    """
    Check if observation point falls in a prism edge parallel to northing

    Return True if the observation point falls in one of the prism edges
    parallel to the northing direction or in any one of the vertices of the
    prism. Return False if otherwise.

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

    Returns
    -------
    result : bool
        Return True if the observation point falls in one of the prism edges
        parallel to the northing direction or in any one of the vertices of the
        prism. Return False if otherwise.
    """
    in_easting = (easting == prism[0]) or (easting == prism[1])
    in_northing = prism[2] <= northing <= prism[3]
    in_upward = (upward == prism[4]) or (upward == prism[5])
    if in_easting and in_northing and in_upward:
        return True
    return False


@jit(nopython=True)
def is_point_on_upward_edge(easting, northing, upward, prism):
    """
    Check if observation point falls in a prism edge parallel to upward

    Return True if the observation point falls in one of the prism edges
    parallel to the upward direction or in any one of the vertices of the
    prism. Return False if otherwise.

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

    Returns
    -------
    result : bool
        Return True if the observation point falls in one of the prism edges
        parallel to the upward direction or in any one of the vertices of the
        prism. Return False if otherwise.
    """
    in_easting = (easting == prism[0]) or (easting == prism[1])
    in_northing = (northing == prism[2]) or (northing == prism[3])
    in_upward = prism[4] <= upward <= prism[5]
    if in_easting and in_northing and in_upward:
        return True
    return False


@jit(nopython=True)
def is_point_on_east_face(easting, northing, upward, prism):
    """
    Check if observation point falls in the eastern face of the prism

    Return True if the observation point falls in the eastern face of the
    prism, without including the vertices or the edges (the inside of the
    face).

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

    Returns
    -------
    result : bool
        Return True if the observation point falls inside the eastern face of
        the prism. Return False if otherwise.
    """
    on_east_face = (
        (easting == prism[1])
        and (prism[2] < northing < prism[3])
        and (prism[4] < upward < prism[5])
    )
    if on_east_face:
        return True
    return False


@jit(nopython=True)
def is_point_on_north_face(easting, northing, upward, prism):
    """
    Check if observation point falls in the northern face of the prism

    Return True if the observation point falls in the northern face of the
    prism, without including the vertices or the edges (the inside of the
    face).

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

    Returns
    -------
    result : bool
        Return True if the observation point falls inside the northern face of
        the prism. Return False if otherwise.
    """
    on_north_face = (
        (prism[0] < easting < prism[1])
        and (northing == prism[3])
        and (prism[4] < upward < prism[5])
    )
    if on_north_face:
        return True
    return False


@jit(nopython=True)
def is_point_on_top_face(easting, northing, upward, prism):
    """
    Check if observation point falls in the top face of the prism

    Return True if the observation point falls in the top face of the prism,
    without including the vertices or the edges (the inside of the face).

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

    Returns
    -------
    result : bool
        Return True if the observation point falls inside the top face of the
        prism. Return False if otherwise.
    """
    on_top_face = (
        (prism[0] < easting < prism[1])
        and (prism[2] < northing < prism[3])
        and (upward == prism[5])
    )
    if on_top_face:
        return True
    return False
