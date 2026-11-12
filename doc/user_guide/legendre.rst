Associated Legendre functions
=============================

The associated Legendre functions (ALFs) :math:`P_n^m(x)` are often used in
spherical harmonic analysis. They are solutions to Legendre's differential
equation

.. math::

    \frac{d}{d x} \left[ \left(1 - x^2\right) \frac{d}{d x} P_n^m(x) \right]
    + \left[ n (n + 1) - \frac{m^2}{1 - x^2} \right] P_n^m(x) = 0,

which has non-trivial and non-singular solutions in the [-1, 1] range when
:math:`n` (the degree) and :math:`m` (the order) are integers and :math:`0 \le m
\le n`. It's common in spherical harmonics for the Legendre functions receive
the cossine of the colatitude :math:`\theta` as input

.. math::

    P_n^m(x) = P_n^m(\cos\theta).

Choclo provides highly optimized implementations of the ALFs and their
derivatives, with two different normalization factors. Our functions employ
techniques from [Holmes2002]_ to make them accurate at least to a maximum degree
of 2800.

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt
    from choclo import legendre

Let's calculate and plot different versions of the ALFs up to degree 3 for the
full range of :math:`\theta`:

.. jupyter-execute::

    theta = np.linspace(0, 180, 50)
    x = np.cos(np.radians(theta))
    max_degree = 3

Since our function calculates :math:`P_n^m(x)` for a single :math:`x`, we can
make a 3D array with the first dimension representing x and fill it out in
a loop:

.. jupyter-execute::

    p = np.zeros((x.size, max_degree + 1, max_degree + 1))
    for i, value in enumerate(x):
        legendre.associated_function_schmidt(value, max_degree, p[i])

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

Notice how the amplitude of the function does not increase greatly with the
degree :math:`n`. This is why we recommend using the normalized functions
instead of the regular ones.
