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

    Examples
    --------
    Let's calculate and plot the function up to degree 3:

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        from choclo import legendre

        x = np.linspace(-1, 1, 50)
        max_degree = 3

    Since our function calculates :math:`P_n^m` for a single :math:`x`, we can
    make a 3D array with the first dimension representing x and fill it out in
    a loop:

    .. jupyter-execute::

        p = np.zeros((x.size, max_degree + 1, max_degree + 1))
        for i, value in enumerate(x):
            legendre.associated_function(value, max_degree, p[i])

    Now we can plot the functions by slicing them at the desired degree and
    order:

    .. jupyter-execute::

        plt.figure(figsize=(8, 4), layout="constrained")
        for n in range(max_degree + 1):
            for m in range(n + 1):
                plt.plot(x, p[:, n, m], label=f"$P_{n}^{m}$")
        plt.legend(ncols=3)
        plt.xlabel("x")
        plt.xlim(-1, 1)
        maxabs = np.abs(p).max()
        plt.ylim(-maxabs, maxabs)
        plt.grid()
        plt.show()

    Notice how the amplitude of the function tends to increase with the degree
    :math:`n`. This is why we recommend using the normalized functions instead.
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
    \le m \le n`. Implements the recursive relations defined in [Alken2022]_,
    which use calculated values of :math:`P_n^m` to calculate its derivatives.
    Since the calculations are recursive, it's not possible to calculate only
    the values for a specific degree and order. All values up to the desired
    degree are always calculated.

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
        are ignored.
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

    Examples
    --------
    Let's calculate and plot the derivative up to degree 3:

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        from choclo import legendre

        theta = np.linspace(0, 180, 50)
        x = np.cos(np.radians(theta))
        max_degree = 3

    To calculate the derivative, we need to first calculate the :math:`P_n^m`.
    Since our function calculates :math:`P_n^m` and its derivative for a single
    :math:`x`, we can make a 3D array with the first dimension representing x
    and fill it out in a loop:

    .. jupyter-execute::

        p = np.zeros((x.size, max_degree + 1, max_degree + 1))
        dp = np.zeros((x.size, max_degree + 1, max_degree + 1))
        for i, value in enumerate(x):
            legendre.associated_function(value, max_degree, p[i])
            legendre.associated_function_derivative(max_degree, p[i], dp[i])

    Now we can plot the derivatives by slicing them at the desired degree and
    order:

    .. jupyter-execute::

        plt.figure(figsize=(8, 4), layout="constrained")
        for n in range(max_degree + 1):
            for m in range(n + 1):
                plt.plot(theta, dp[:, n, m], label=f"$P'_{n}^{m}$")
        plt.legend(ncols=3)
        plt.xlabel("theta (degrees)")
        plt.xlim(0, 180)
        maxabs = np.abs(dp).max()
        plt.ylim(-maxabs, maxabs)
        plt.grid()
        plt.show()

    Notice how the amplitude of the function tends to increase with the degree
    :math:`n`. This is why we recommend using the normalized functions instead.
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
    """
    Schmidt-normalized associated Legendre functions up to a maximum degree.

    The associated Legendre functions functions :math:`P_n^m(x)` are solutions
    to Legendre's equation. Here, :math:`n` is the degree and :math:`m` is the
    order. The Schmidt normalization is applied to the functions to constrain
    their value range and avoid overflows. This normalization if often used in
    geomagnetic field models.

    We calculate their values using the recursive relations defined in
    [Alken2022]_. Hence, it's not possible to calculate only the values for a
    specific degree and order. All values up to the desired degree are always
    calculated.

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
    Does not implement the Condon-Shortly phase element.

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
    Theta derivatives of Schmidt normalized associated Legendre functions.

    Calculates the derivative:

    .. math::

        \\dfrac{\\partial P_n^m}{\\partial \\theta}(\\cos \\theta)

    using the recursive relations defined in [Alken2022]_.

    Higher-order derivatives can be calculated by passing the output of this
    function as the ``p`` argument.

    .. note::

        This function does not include the Condon-Shortly phase.

    .. note::

        First and second order derivatives are tested against the Legendre
        differential equation and produce accurate results until degree and
        order 2800.

    Parameters
    ----------
    max_degree : int
        The maximum degree for the calculation.
    p : numpy.ndarray
        A 2D array with the Schmidt normalized associated Legendre functions
        calculated for :math:`\\cos\\theta`.
    dp : numpy.array
        An 2D array with shape ``(max_degree + 1, max_degree + 1)`` that will
        be filled with the output values.

    References
    ----------
    Alken, Patrick (2022). GSL Technical Report #1 - GSL-TR-001-20220827 -
      Implementation of associated Legendre functions in GSL.
      https://www.gnu.org/software/gsl/tr/tr001.pdf
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
    """
    Fully normalized associated Legendre functions up to maximum degree.

    The functions :math:`P_n^m` are solutions to Legendre's equation. The full
    normalization is applied to the functions to constrain their value range.
    This normalization if often used in gravity field models. We calculate
    their values using the recursive relations defined in [Alken2022]_ but with
    the definition of the normalization from Hofmann-Wellenhof and Moritz
    (2006).

    .. note::

        This function does not include the Condon-Shortly phase.

    Parameters
    ----------
    x : float
        The argument of :math:`P_n^m(x)`. Must be in the range [-1, 1].
    max_degree : int
        The maximum degree for the calculation.
    p : numpy.array
        A 2D array with shape ``(max_degree + 1, max_degree + 1)`` that will be
        filled with the output values.

    References
    ----------
    Alken, Patrick (2022). GSL Technical Report #1 - GSL-TR-001-20220827 -
      Implementation of associated Legendre functions in GSL.
      https://www.gnu.org/software/gsl/tr/tr001.pdf
    Hofmann-Wellenhof, B., & Moritz, H. (2006). Physical Geodesy (2nd, corr.
      ed. 2006 edition ed.). Wien; New York: Springer.
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
    Theta derivative of fully normalized associated Legendre functions.

    Calculates the derivative:

    .. math::

        \\dfrac{\\partial P_n^m}{\\partial \\theta}(\\cos \\theta)

    Uses the recursive relations defined in [Alken2022]_ but with the
    definition of the normalization from Hofmann-Wellenhof and Moritz (2006).

    Higher-order derivatives can be calculated by passing the output of this
    function as the ``p`` argument.

    .. note::

        This function does not include the Condon-Shortly phase.

    Parameters
    ----------
    max_degree : int
        The maximum degree for the calculation.
    p : numpy.ndarray
        A 2D array with the fully normalized associated Legendre functions
        calculated for :math:`\\cos\\theta`.
    dp : numpy.array
        A 2D array with shape ``(max_degree + 1, max_degree + 1)`` that will be
        filled with the output values.

    References
    ----------
    [Alken2022]_
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
