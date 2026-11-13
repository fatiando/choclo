# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Calculation of associated Legendre functions and their derivatives.
"""

import numba
import numpy as np


@numba.jit(nopython=True)
def _rescale(u, max_degree, p):
    r"""
    Rescale Legendre functions to their original range.

    Parameters
    ----------
    u : float
        :math:`u = \sqrt{(1 - x)(1 + x)}`.
    max_degree : int
        The maximum degree :math:`n` for the calculation.
    p : numpy.array
        A 2D array with shape ``(max_degree + 1, max_degree + 1)`` that will be
        filled in place with the output values. The rows of the array correspond
        to the degree :math:`n` and the columns to the order :math:`m`. Thus,
        ``p[n, m]`` corresponds to :math:`P_n^m(x)`. Elements where ``m > n``
        are not filled out and will retain their original values.
    """
    rescale = 1e280
    for m in range(max_degree + 1):
        if m > 0:
            rescale *= u
        for n in range(m, max_degree + 1):
            p[n, m] *= rescale


@numba.jit(nopython=True)
def associated_function(x, max_degree, p):
    r"""
    Associated Legendre functions up to a maximum degree.

    The associated Legendre functions :math:`P_n^m(x)` are solutions to
    Legendre's equation, with :math:`n` (the degree) and :math:`m` (the order)
    being integers and :math:`0 \le m \le n`. We calculate their values using
    the recursive relations defined in [Alken2022]_. Being recursive, it's not
    possible to calculate only the values for a specific degree and order. All
    values up to the desired degree are always calculated.

    The output is filled in place in the parameter ``p`` so that the users can
    control the memory allocation. In spherical harmonics, this function if
    often called in a loop and so reusing memory can save computation time.

    .. warning::

        This function does not implement normalization and should not
        be used beyond degree 13 because they will overflow 64-bit float
        range. Use :func:`choclo.legendre.associated_function_schmidt` and
        :func:`choclo.legendre.associated_function_fully` instead.

    Parameters
    ----------
    x : float
        The argument of :math:`P_n^m(x)`. Must be in the range [-1, 1].
    max_degree : int
        The maximum degree :math:`n` for the calculation.
    p : numpy.array
        A 2D array with shape ``(max_degree + 1, max_degree + 1)`` that will be
        filled in place with the output values. The rows of the array correspond
        to the degree :math:`n` and the columns to the order :math:`m`. Thus,
        ``p[n, m]`` corresponds to :math:`P_n^m(x)`. Elements where ``m > n``
        are not filled out and will retain their original values.

    Notes
    -----
    The functions :math:`P_n^m(x)` are solutions to Legendre's differential
    equation

    .. math::

        \frac{d}{d x} \left[ \left(1 - x^2\right) \frac{d}{d x} P_n^m(x) \right]
        + \left[ n (n + 1) - \frac{m^2}{1 - x^2} \right] P_n^m(x) = 0,

    which has non-trivial and non-singular solutions in the [-1, 1] range when
    :math:`n` and :math:`m` are integers and :math:`0 \le m \le n`.

    .. note::

        Does not implement the Condon-Shortly phase element.

    References
    ----------
    [Alken2022]_
    """
    u = np.sqrt((1 - x) * (1 + x))
    p[0, 0] = 1
    for n in range(1, max_degree + 1):
        for m in range(n - 1):
            a_nm = (2 * n - 1) / (n - m)
            b_nm = -(n + m - 1) / (n - m)
            p[n, m] = a_nm * x * p[n - 1, m] + b_nm * p[n - 2, m]
        c_nm = 2 * n - 1
        p[n, n - 1] = c_nm * x * p[n - 1, n - 1]
        d_nm = 2 * n - 1
        p[n, n] = d_nm * u * p[n - 1, n - 1]


@numba.jit(nopython=True)
def associated_function_derivative(max_degree, p, dp):
    r"""
    Theta derivative of the associated Legendre functions.

    Calculates the derivative

    .. math::

        \dfrac{\partial P_n^m}{\partial \theta}(\cos \theta)

    of the associated Legendre function :math:`P_n^m(\cos\theta)`, with
    :math:`n` (the degree) and :math:`m` (the order) being integers and :math:`0
    \le m \le n`.

    Implements the recursive relations defined in [Alken2022]_, which use
    calculated values of :math:`P_n^m` to calculate its derivatives. Since the
    calculations are recursive, it's not possible to calculate only the values
    for a specific degree and order. All values up to the desired degree are
    always calculated.

    Higher-order derivatives can be calculated by passing the output of this
    function as the ``p`` argument in a subsequent call.

    The output is filled in place in the parameter ``dp`` so that the users
    can control the memory allocation. In spherical harmonics, this function if
    often called in a loop and so reusing memory can save computation time.

    .. warning::

        This function does not implement normalization and should not be used
        beyond degree 13 because they will overflow 64-bit float range. Use
        :func:`choclo.legendre.associated_function_schmidt_derivative` and
        :func:`choclo.legendre.associated_function_fully_derivative` instead.

    Parameters
    ----------
    max_degree : int
        The maximum degree :math:`n` for the calculation.
    p : numpy.ndarray
        A 2D array with shape ``(max_degree + 1, max_degree + 1)`` with values
        of :math:`P_n^m(\cos\theta)`. The rows of the array correspond to the
        degree :math:`n` and the columns to the order :math:`m`. Thus, ``p[n,
        m]`` corresponds to :math:`P_n^m(\cos\theta)`. Elements where ``m > n``
        are ignored. See :func:`choclo.legendre.associated_function`.
    dp : numpy.array
        A 2D array with shape ``(max_degree + 1, max_degree + 1)`` that will be
        filled in place with the output values. The rows of the array correspond
        to the degree :math:`n` and the columns to the order :math:`m`. Thus,
        ``dp[n, m]`` corresponds to
        :math:`\partial P_n^m(\cos\theta) / \partial\theta`. Elements
        where ``m > n`` are not filled out and will retain their original
        values.

    Notes
    -----
    The functions :math:`P_n^m(x)` are solutions to Legendre's differential
    equation

    .. math::

        \frac{d}{d x} \left[ \left(1 - x^2\right) \frac{d}{d x} P_n^m(x) \right]
        + \left[ n (n + 1) - \frac{m^2}{1 - x^2} \right] P_n^m(x) = 0,

    which has non-trivial and non-singular solutions in the [-1, 1] range when
    :math:`n` and :math:`m` are integers and :math:`0 \le m \le n`.

    In spherical harmonics, these functions are usually evaluated as
    :math:`P_n^m(\cos\theta)`, in which :math:`\theta` is the geocentric
    colatitude with range [0, 180] degrees. It's often required to know the
    derivative of the :math:`P_n^m` with respect to :math:`\theta`, which is
    what this function calculates.

    .. note::

        Does not implement the Condon-Shortly phase element.

    References
    ----------
    [Alken2022]_
    """
    dp[0, 0] = 0
    for n in range(1, max_degree + 1):
        a_nm = -1
        dp[n, 0] = a_nm * p[n, 1]
        for m in range(1, n):
            b_nm = 0.5 * (n + m) * (n - m + 1)
            c_nm = -0.5
            dp[n, m] = b_nm * p[n, m - 1] + c_nm * p[n, m + 1]
        d_nm = n
        dp[n, n] = d_nm * p[n, n - 1]


@numba.jit(nopython=True)
def associated_function_schmidt(x, max_degree, p):
    r"""
    Schmidt-normalized associated Legendre functions up to a maximum degree.

    The associated Legendre functions functions :math:`P_n^m(x)` are solutions
    to Legendre's equation, with :math:`n` (the degree) and :math:`m` (the
    order) being integers and :math:`0 \le m \le n`. The Schmidt normalization
    is applied to the functions to constrain their value range and avoid
    overflows. This normalization if often used in geomagnetic field models.

    We calculate their values using the recursive relations defined in
    [Alken2022]_. Being recursive, it's not possible to calculate only the
    values for a specific degree and order. All values up to the desired degree
    are always calculated.

    Parameters
    ----------
    x : float
        The argument of :math:`P_n^m(x)`. Must be in the range [-1, 1].
    max_degree : int
        The maximum degree :math:`n` for the calculation.
    p : numpy.array
        A 2D array with shape ``(max_degree + 1, max_degree + 1)`` that will be
        filled in place with the output values. The rows of the array correspond
        to the degree :math:`n` and the columns to the order :math:`m`. Thus,
        ``p[n, m]`` corresponds to :math:`P_n^m(x)`. Elements where ``m > n``
        are not filled out and will retain their original values.

    Notes
    -----
    The functions :math:`P_n^m(x)` are solutions to Legendre's differential
    equation

    .. math::

        \frac{d}{d x} \left[ \left(1 - x^2\right) \frac{d}{d x} P_n^m(x) \right]
        + \left[ n (n + 1) - \frac{m^2}{1 - x^2} \right] P_n^m(x) = 0,

    which has non-trivial and non-singular solutions in the [-1, 1] range when
    :math:`n` and :math:`m` are integers and :math:`0 \le m \le n`.

    In practice, their amplitudes tend to increase dramatically with the degree
    :math:`n`, and so normalization techniques are often employed to keep
    amplitudes to reasonable values. In geomagnetism, it's common to employ
    Schmidt normalization

    .. math::

        S_n^m = \sqrt{(2 - \delta_{m,0})\dfrac{(n - m)!}{(n + m)!}},

    in which :math:`\delta_{m,0}` is the `Kronecker delta function
    <https://en.wikipedia.org/wiki/Kronecker_delta>`__. The normalization factor
    :math:`S_n^m` multiplies the :math:`P_n^m`. In practice, the equation above
    is not used. Instead, the Legendre function is evaluated using recursive
    equations specifically for the normalized version.

    .. note::

        Does not implement the Condon-Shortly phase element.

    .. note::

        This function uses the scaling scheme of [Holmes2002]_ and produces accurate
        results until degree and order 2800.

    References
    ----------
    [Alken2022]_
    [Holmes2002]_
    """
    u = np.sqrt((1 - x) * (1 + x))
    # Pre-compute square roots of integers used in the loops
    sqrt = np.sqrt(np.arange(2 * (max_degree + 1)))
    # Use the Holmes and Featherstone (2002) scaling to compute scaled Pnm
    # All terms are scaled by the max float range 1e280
    p[0, 0] = 1 * 1e-280
    p[1, 0] = x * 1e-280
    # This equation doesn't have u because of the scaling
    p[1, 1] = p[0, 0]
    # Calculate the zero order terms first
    for n in range(2, max_degree + 1):
        a_n0 = (2 * n - 1) / (sqrt[n] ** 2)
        b_n0 = -((sqrt[n - 1] / sqrt[n]) ** 2)
        p[n, 0] = a_n0 * x * p[n - 1, 0] + b_n0 * p[n - 2, 0]
    # Now calculate the other terms
    for n in range(2, max_degree + 1):
        for m in range(1, n - 1):
            a_nm = (2 * n - 1) / (sqrt[n + m] * sqrt[n - m])
            b_nm = -sqrt[n + m - 1] * sqrt[n - m - 1] / (sqrt[n + m] * sqrt[n - m])
            p[n, m] = a_nm * x * p[n - 1, m] + b_nm * p[n - 2, m]
        c_nm = sqrt[2 * n - 1]
        p[n, n - 1] = c_nm * x * p[n - 1, n - 1]
        d_nm = sqrt[2 * n - 1] / sqrt[2 * n]
        # This equation doesn't have u because of the scaling
        p[n, n] = d_nm * p[n - 1, n - 1]
    # Now return everything to the original float range and rescale by u**m
    _rescale(u, max_degree, p)


@numba.jit(nopython=True)
def associated_function_schmidt_derivative(max_degree, p, dp):
    r"""
    Theta derivatives of the Schmidt-normalized associated Legendre functions.

    Calculates the derivative

    .. math::

        \dfrac{\partial P_n^m}{\partial\theta}(\cos\theta),

    of the associated Legendre function :math:`P_n^m(\cos\theta)`, with
    :math:`n` (the degree) and :math:`m` (the order) being integers and :math:`0
    \le m \le n`. Assumes that the Legendre function is Schmidt-normalized.

    Implements the recursive relations defined in [Alken2022]_, which use
    calculated values of :math:`P_n^m` to calculate its derivatives. Since the
    calculations are recursive, it's not possible to calculate only the values
    for a specific degree and order. All values up to the desired degree are
    always calculated.

    Higher-order derivatives can be calculated by passing the output of this
    function as the ``p`` argument in a subsequent call.

    The output is filled in place in the parameter ``dp`` so that the users
    can control the memory allocation. In spherical harmonics, this function if
    often called in a loop and so reusing memory can save computation time.

    Parameters
    ----------
    max_degree : int
        The maximum degree :math:`n` for the calculation.
    p : numpy.ndarray
        A 2D array with shape ``(max_degree + 1, max_degree + 1)`` with
        values of the Schmidt-normalized :math:`P_n^m(\cos\theta)`. The
        rows of the array correspond to the degree :math:`n` and the
        columns to the order :math:`m`. Thus, ``p[n, m]`` corresponds to
        :math:`P_n^m(\cos\theta)`. Elements where ``m > n`` are ignored. See
        :func:`choclo.legendre.associated_function_schmidt`.
    dp : numpy.array
        A 2D array with shape ``(max_degree + 1, max_degree + 1)`` that will be
        filled in place with the output values. The rows of the array correspond
        to the degree :math:`n` and the columns to the order :math:`m`. Thus,
        ``dp[n, m]`` corresponds to
        :math:`\partial P_n^m(\cos\theta) / \partial\theta`. Elements
        where ``m > n`` are not filled out and will retain their original
        values.

    Notes
    -----
    The functions :math:`P_n^m(x)` are solutions to Legendre's differential
    equation

    .. math::

        \frac{d}{d x} \left[ \left(1 - x^2\right) \frac{d}{d x} P_n^m(x) \right]
        + \left[ n (n + 1) - \frac{m^2}{1 - x^2} \right] P_n^m(x) = 0,

    which has non-trivial and non-singular solutions in the [-1, 1] range when
    :math:`n` and :math:`m` are integers and :math:`0 \le m \le n`.

    In spherical harmonics, these functions are usually evaluated as
    :math:`P_n^m(\cos\theta)`, in which :math:`\theta` is the geocentric
    colatitude with range [0, 180] degrees. It's often required to know the
    derivative of the :math:`P_n^m` with respect to :math:`\theta`, which is
    what this function calculates.

    In practice, their amplitudes tend to increase dramatically with the degree
    :math:`n`, and so normalization techniques are often employed to keep
    amplitudes to reasonable values. In geomagnetism, it's common to employ
    Schmidt normalization

    .. math::

        S_n^m = \sqrt{(2 - \delta_{m,0})\dfrac{(n - m)!}{(n + m)!}},

    in which :math:`\delta_{m,0}` is the `Kronecker delta function
    <https://en.wikipedia.org/wiki/Kronecker_delta>`__. In practice, the
    equation above is not used. Instead, the derivative is evaluated using
    recursive equations specifically for the normalized version.

    .. note::

        Does not implement the Condon-Shortly phase element.

    .. note::

        First and second order derivatives are tested against the Legendre
        differential equation and produce accurate results until degree and
        order 2800. Higher order derivatives have not been tested.

    References
    ----------
    [Alken2022]_
    """
    # Pre-compute square roots of integers used in the loops
    sqrt = np.sqrt(np.arange(2 * (max_degree + 1)))
    dp[0, 0] = 0
    dp[1, 0] = -p[1, 1]
    dp[1, 1] = p[1, 0]
    for n in range(2, max_degree + 1):
        a_nm = -sqrt[n] * sqrt[n + 1] / sqrt[2]
        dp[n, 0] = a_nm * p[n, 1]
        b_nm = 0.5 * sqrt[n] * sqrt[n + 1] * sqrt[2]
        c_nm = -0.5 * sqrt[n + 2] * sqrt[n - 1]
        dp[n, 1] = b_nm * p[n, 0] + c_nm * p[n, 2]
        for m in range(2, n):
            b_nm = 0.5 * sqrt[n + m] * sqrt[n - m + 1]
            c_nm = -0.5 * sqrt[n + m + 1] * sqrt[n - m]
            dp[n, m] = b_nm * p[n, m - 1] + c_nm * p[n, m + 1]
        d_nm = 0.5 * sqrt[2] * sqrt[n]
        dp[n, n] = d_nm * p[n, n - 1]


@numba.jit(nopython=True)
def associated_function_fully(x, max_degree, p):
    r"""
    Fully-normalized associated Legendre functions up to maximum degree.

    The associated Legendre functions functions :math:`P_n^m(x)` are solutions
    to Legendre's equation, with :math:`n` (the degree) and :math:`m` (the
    order) being integers and :math:`0 \le m \le n`. The full normalization
    is applied to the functions to constrain their value range and avoid
    overflows. This normalization if often used in gravity field models.

    We calculate their values using the recursive relations defined
    in [Alken2022]_ but with the definition of the normalization from
    [Hofmann-Wellenhof2006]_. Being recursive, it's not possible to calculate
    only the values for a specific degree and order. All values up to the
    desired degree are always calculated.

    Parameters
    ----------
    x : float
        The argument of :math:`P_n^m(x)`. Must be in the range [-1, 1].
    max_degree : int
        The maximum degree :math:`n` for the calculation.
    p : numpy.array
        A 2D array with shape ``(max_degree + 1, max_degree + 1)`` that will be
        filled in place with the output values. The rows of the array correspond
        to the degree :math:`n` and the columns to the order :math:`m`. Thus,
        ``p[n, m]`` corresponds to :math:`P_n^m(x)`. Elements where ``m > n``
        are not filled out and will retain their original values.

    Notes
    -----
    The functions :math:`P_n^m(x)` are solutions to Legendre's differential
    equation

    .. math::

        \frac{d}{d x} \left[ \left(1 - x^2\right) \frac{d}{d x} P_n^m(x) \right]
        + \left[ n (n + 1) - \frac{m^2}{1 - x^2} \right] P_n^m(x) = 0,

    which has non-trivial and non-singular solutions in the [-1, 1] range when
    :math:`n` and :math:`m` are integers and :math:`0 \le m \le n`.

    In practice, their amplitudes tend to increase dramatically with the degree
    :math:`n`, and so normalization techniques are often employed to keep
    amplitudes to reasonable values. In gravimetry and geodesy, it's common to
    employ full normalization

    .. math::

        F_n^m = \sqrt{2(2n + 1)\dfrac{(n - m)!}{(n + m)!}},

    The normalization factor :math:`F_n^m` multiplies the :math:`P_n^m`. In
    practice, the equation above is not used. Instead, the Legendre function is
    evaluated using recursive equations specifically for the normalized version.

    .. note::

        Does not implement the Condon-Shortly phase element.

    .. note::

        This function uses the scaling scheme of [Holmes2002]_ and produces accurate
        results until degree and order 2800.

    References
    ----------
    [Alken2022]_
    [Holmes2002]_
    [Hofmann-Wellenhof2006]_
    """
    u = np.sqrt((1 - x) * (1 + x))
    # Pre-compute square roots of integers used in the loops
    sqrt = np.sqrt(np.arange(2 * (max_degree + 1)))
    # Use the Holmes and Featherstone (2002) scaling to compute scaled Pnm
    # All terms are scaled by the max float range 1e280
    p[0, 0] = 1 * 1e-280
    p[1, 0] = x * sqrt[3] * 1e-280
    # This equation doesn't have u because of the scaling
    p[1, 1] = p[0, 0] * sqrt[3]
    # Calculate the zero order terms first
    for n in range(2, max_degree + 1):
        a_n0 = sqrt[2 * n - 1] * sqrt[2 * n + 1] / n
        b_n0 = -(n - 1) * sqrt[2 * n + 1] / (n * sqrt[2 * n - 3])
        p[n, 0] = a_n0 * x * p[n - 1, 0] + b_n0 * p[n - 2, 0]
    # Now calculate the other terms
    for n in range(2, max_degree + 1):
        for m in range(1, n - 1):
            a_nm = sqrt[2 * n + 1] * sqrt[2 * n - 1] / (sqrt[n + m] * sqrt[n - m])
            b_nm = (
                -sqrt[n + m - 1]
                * sqrt[n - m - 1]
                * sqrt[2 * n + 1]
                / (sqrt[n + m] * sqrt[n - m] * sqrt[2 * n - 3])
            )
            p[n, m] = a_nm * x * p[n - 1, m] + b_nm * p[n - 2, m]
        c_nm = sqrt[2 * n + 1]
        p[n, n - 1] = c_nm * x * p[n - 1, n - 1]
        d_nm = sqrt[2 * n + 1] / sqrt[2 * n]
        # This equation doesn't have u because of the scaling
        p[n, n] = d_nm * p[n - 1, n - 1]
    # Now return everything to the original float range and rescale by u**m
    _rescale(u, max_degree, p)


@numba.jit(nopython=True)
def associated_function_fully_derivative(max_degree, p, dp):
    r"""
    Theta derivative of the fully-normalized associated Legendre functions.

    Calculates the derivative

    .. math::

        \dfrac{\partial P_n^m}{\partial\theta}(\cos\theta),

    of the associated Legendre function :math:`P_n^m(\cos\theta)`, with
    :math:`n` (the degree) and :math:`m` (the order) being integers and :math:`0
    \le m \le n`. Assumes that the Legendre function is fully-normalized.

    Implements the recursive relations defined in [Alken2022]_ but with the
    definition of the normalization from [Hofmann-Wellenhof2006]_, which use
    calculated values of :math:`P_n^m` to calculate its derivatives. Since the
    calculations are recursive, it's not possible to calculate only the values
    for a specific degree and order. All values up to the desired degree are
    always calculated.

    Higher-order derivatives can be calculated by passing the output of this
    function as the ``p`` argument in a subsequent call.

    The output is filled in place in the parameter ``dp`` so that the users
    can control the memory allocation. In spherical harmonics, this function if
    often called in a loop and so reusing memory can save computation time.

    Parameters
    ----------
    max_degree : int
        The maximum degree :math:`n` for the calculation.
    p : numpy.ndarray
        A 2D array with shape ``(max_degree + 1, max_degree + 1)`` with
        values of the fully-normalized :math:`P_n^m(\cos\theta)`. The
        rows of the array correspond to the degree :math:`n` and the
        columns to the order :math:`m`. Thus, ``p[n, m]`` corresponds to
        :math:`P_n^m(\cos\theta)`. Elements where ``m > n`` are ignored. See
        :func:`choclo.legendre.associated_function_fully`.
    dp : numpy.array
        A 2D array with shape ``(max_degree + 1, max_degree + 1)`` that will be
        filled in place with the output values. The rows of the array correspond
        to the degree :math:`n` and the columns to the order :math:`m`. Thus,
        ``dp[n, m]`` corresponds to
        :math:`\partial P_n^m(\cos\theta) / \partial\theta`. Elements
        where ``m > n`` are not filled out and will retain their original
        values.

    Notes
    -----
    The functions :math:`P_n^m(x)` are solutions to Legendre's differential
    equation

    .. math::

        \frac{d}{d x} \left[ \left(1 - x^2\right) \frac{d}{d x} P_n^m(x) \right]
        + \left[ n (n + 1) - \frac{m^2}{1 - x^2} \right] P_n^m(x) = 0,

    which has non-trivial and non-singular solutions in the [-1, 1] range when
    :math:`n` and :math:`m` are integers and :math:`0 \le m \le n`.

    In spherical harmonics, these functions are usually evaluated as
    :math:`P_n^m(\cos\theta)`, in which :math:`\theta` is the geocentric
    colatitude with range [0, 180] degrees. It's often required to know the
    derivative of the :math:`P_n^m` with respect to :math:`\theta`, which is
    what this function calculates.

    In practice, their amplitudes tend to increase dramatically with the degree
    :math:`n`, and so normalization techniques are often employed to keep
    amplitudes to reasonable values. In gravimetry and geodesy, it's common to
    employ full normalization

    .. math::

        F_n^m = \sqrt{2(2n + 1)\dfrac{(n - m)!}{(n + m)!}},

    The normalization factor :math:`F_n^m` multiplies the :math:`P_n^m`.
    In practice, the equation above is not used. Instead, the derivative is
    evaluated using recursive equations specifically for the normalized version.

    .. note::

        Does not implement the Condon-Shortly phase element.

    .. note::

        First and second order derivatives are tested against the Legendre
        differential equation and produce accurate results until degree and
        order 2800. Higher order derivatives have not been tested.

    References
    ----------
    [Alken2022]_
    [Holmes2002]_
    [Hofmann-Wellenhof2006]_
    """
    # Pre-compute square roots of integers used in the loops
    sqrt = np.sqrt(np.arange(2 * (max_degree + 1)))
    dp[0, 0] = 0
    dp[1, 0] = -p[1, 1]
    dp[1, 1] = p[1, 0]
    for n in range(2, max_degree + 1):
        a_nm = -sqrt[n] * sqrt[n + 1] / sqrt[2]
        dp[n, 0] = a_nm * p[n, 1]
        b_n1 = 0.5 * sqrt[2] * sqrt[n + 1] * sqrt[n]
        c_n1 = -0.5 * sqrt[n + 2] * sqrt[n - 1]
        dp[n, 1] = b_n1 * p[n, 0] + c_n1 * p[n, 2]
        for m in range(2, n):
            b_nm = 0.5 * sqrt[n + m] * sqrt[n - m + 1]
            c_nm = -0.5 * sqrt[n + m + 1] * sqrt[n - m]
            dp[n, m] = b_nm * p[n, m - 1] + c_nm * p[n, m + 1]
        d_nm = 0.5 * sqrt[2] * sqrt[n]
        dp[n, n] = d_nm * p[n, n - 1]
