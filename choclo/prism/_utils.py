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
def is_interior_point(
    easting,
    northing,
    upward,
    prism_west,
    prism_east,
    prism_south,
    prism_north,
    prism_bottom,
    prism_top,
):
    """
    Check if observation point falls inside the prism

    Return True if the observation point falls inside the prism,
    not including vertices, edges or faces.
    Return False if otherwise.

    Parameters
    ----------
    easting, northing, upward : float
        Easting, northing and upward coordinates of the observation point. Must
        be in meters.
    prism_west, prism_east, prism_south, prism_north, prism_bottom, prism_top : float
        The boundaries of the prism. Must be in meters.

    Returns
    -------
    result : bool
        Return True if the observation point falls inside the prism, not
        including vertices, edges or faces. Return False if otherwise.
    """
    in_easting = prism_west < easting < prism_east
    in_northing = prism_south < northing < prism_north
    in_upward = prism_bottom < upward < prism_top
    return in_easting and in_northing and in_upward


@jit(nopython=True)
def is_point_on_edge(
    easting,
    northing,
    upward,
    prism_west,
    prism_east,
    prism_south,
    prism_north,
    prism_bottom,
    prism_top,
):
    """
    Check if observation point falls on any edge of the prism

    Return True if the observation point falls in any one of the prism edges,
    including the prism vertices.
    Return False if otherwise.

    Parameters
    ----------
    easting, northing, upward : float
        Easting, northing and upward coordinates of the observation point. Must
        be in meters.
    prism_west, prism_east, prism_south, prism_north, prism_bottom, prism_top : float
        The boundaries of the prism. Must be in meters.

    Returns
    -------
    result : bool
        Return True if the observation point falls in any one of the prism
        edges. Return False if otherwise.
    """
    result = (
        is_point_on_easting_edge(
            easting,
            northing,
            upward,
            prism_west,
            prism_east,
            prism_south,
            prism_north,
            prism_bottom,
            prism_top,
        )
        or is_point_on_northing_edge(
            easting,
            northing,
            upward,
            prism_west,
            prism_east,
            prism_south,
            prism_north,
            prism_bottom,
            prism_top,
        )
        or is_point_on_upward_edge(
            easting,
            northing,
            upward,
            prism_west,
            prism_east,
            prism_south,
            prism_north,
            prism_bottom,
            prism_top,
        )
    )
    return result


@jit(nopython=True)
def is_point_on_easting_edge(
    easting,
    northing,
    upward,
    prism_west,
    prism_east,
    prism_south,
    prism_north,
    prism_bottom,
    prism_top,
):
    """
    Check if observation point falls in a prism edge parallel to easting

    Return True if the observation point falls in one of the prism edges
    parallel to the easting direction or in any one of the vertices of the
    prism. Return False if otherwise.

    Parameters
    ----------
    easting, northing, upward : float
        Easting, northing and upward coordinates of the observation point. Must
        be in meters.
    prism_west, prism_east, prism_south, prism_north, prism_bottom, prism_top : float
        The boundaries of the prism. Must be in meters.

    Returns
    -------
    result : bool
        Return True if the observation point falls in one of the prism edges
        parallel to the easting direction or in any one of the vertices of the
        prism. Return False if otherwise.
    """
    in_easting = prism_west <= easting <= prism_east
    in_northing = northing in (prism_south, prism_north)
    in_upward = upward in (prism_bottom, prism_top)
    if in_easting and in_northing and in_upward:
        return True
    return False


@jit(nopython=True)
def is_point_on_northing_edge(
    easting,
    northing,
    upward,
    prism_west,
    prism_east,
    prism_south,
    prism_north,
    prism_bottom,
    prism_top,
):
    """
    Check if observation point falls in a prism edge parallel to northing

    Return True if the observation point falls in one of the prism edges
    parallel to the northing direction or in any one of the vertices of the
    prism. Return False if otherwise.

    Parameters
    ----------
    easting, northing, upward : float
        Easting, northing and upward coordinates of the observation point. Must
        be in meters.
    prism_west, prism_east, prism_south, prism_north, prism_bottom, prism_top : float
        The boundaries of the prism. Must be in meters.

    Returns
    -------
    result : bool
        Return True if the observation point falls in one of the prism edges
        parallel to the northing direction or in any one of the vertices of the
        prism. Return False if otherwise.
    """
    in_easting = easting in (prism_west, prism_east)
    in_northing = prism_south <= northing <= prism_north
    in_upward = upward in (prism_bottom, prism_top)
    if in_easting and in_northing and in_upward:
        return True
    return False


@jit(nopython=True)
def is_point_on_upward_edge(
    easting,
    northing,
    upward,
    prism_west,
    prism_east,
    prism_south,
    prism_north,
    prism_bottom,
    prism_top,
):
    """
    Check if observation point falls in a prism edge parallel to upward

    Return True if the observation point falls in one of the prism edges
    parallel to the upward direction or in any one of the vertices of the
    prism. Return False if otherwise.

    Parameters
    ----------
    easting, northing, upward : float
        Easting, northing and upward coordinates of the observation point. Must
        be in meters.
    prism_west, prism_east, prism_south, prism_north, prism_bottom, prism_top : float
        The boundaries of the prism. Must be in meters.

    Returns
    -------
    result : bool
        Return True if the observation point falls in one of the prism edges
        parallel to the upward direction or in any one of the vertices of the
        prism. Return False if otherwise.
    """
    in_easting = easting in (prism_west, prism_east)
    in_northing = northing in (prism_south, prism_north)
    in_upward = prism_bottom <= upward <= prism_top
    if in_easting and in_northing and in_upward:
        return True
    return False


@jit(nopython=True)
def is_point_on_east_face(
    easting,
    northing,
    upward,
    prism_west,  # noqa: U100
    prism_east,
    prism_south,
    prism_north,
    prism_bottom,
    prism_top,
):
    """
    Check if observation point falls in the eastern face of the prism

    Return True if the observation point falls in the eastern face of the
    prism, without including the vertices or the edges (the inside of the
    face).

    Parameters
    ----------
    easting, northing, upward : float
        Easting, northing and upward coordinates of the observation point. Must
        be in meters.
    prism_west, prism_east, prism_south, prism_north, prism_bottom, prism_top : float
        The boundaries of the prism. Must be in meters.

    Returns
    -------
    result : bool
        Return True if the observation point falls inside the eastern face of
        the prism. Return False if otherwise.
    """
    on_east_face = (
        (easting == prism_east)
        and (prism_south < northing < prism_north)
        and (prism_bottom < upward < prism_top)
    )
    if on_east_face:
        return True
    return False


@jit(nopython=True)
def is_point_on_north_face(
    easting,
    northing,
    upward,
    prism_west,
    prism_east,
    prism_south,  # noqa: U100
    prism_north,
    prism_bottom,
    prism_top,
):
    """
    Check if observation point falls in the northern face of the prism

    Return True if the observation point falls in the northern face of the
    prism, without including the vertices or the edges (the inside of the
    face).

    Parameters
    ----------
    easting, northing, upward : float
        Easting, northing and upward coordinates of the observation point. Must
        be in meters.
    prism_west, prism_east, prism_south, prism_north, prism_bottom, prism_top : float
        The boundaries of the prism. Must be in meters.

    Returns
    -------
    result : bool
        Return True if the observation point falls inside the northern face of
        the prism. Return False if otherwise.
    """
    on_north_face = (
        (prism_west < easting < prism_east)
        and (northing == prism_north)
        and (prism_bottom < upward < prism_top)
    )
    if on_north_face:
        return True
    return False


@jit(nopython=True)
def is_point_on_top_face(
    easting,
    northing,
    upward,
    prism_west,
    prism_east,
    prism_south,
    prism_north,
    prism_bottom,  # noqa: U100
    prism_top,
):
    """
    Check if observation point falls in the top face of the prism

    Return True if the observation point falls in the top face of the prism,
    without including the vertices or the edges (the inside of the face).

    Parameters
    ----------
    easting, northing, upward : float
        Easting, northing and upward coordinates of the observation point. Must
        be in meters.
    prism_west, prism_east, prism_south, prism_north, prism_bottom, prism_top : float
        The boundaries of the prism. Must be in meters.

    Returns
    -------
    result : bool
        Return True if the observation point falls inside the top face of the
        prism. Return False if otherwise.
    """
    on_top_face = (
        (prism_west < easting < prism_east)
        and (prism_south < northing < prism_north)
        and (upward == prism_top)
    )
    if on_top_face:
        return True
    return False
