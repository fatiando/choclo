.. title:: Home

.. raw:: html

    <h1 class="display-1">Choclo</h1>

    <div class="sd-fs-3 sd-mb-4">
    Kernel functions for your geophysical models
    </div>

**Choclo** is a Python library that hosts optimized forward modelling and
kernel functions for running geophysical forward and inverse models, intended
to be used by other libraries as the underlying layer of their computation.

"Choclo" is a term used in some countries of South America to refer to corn,
originated from the `quechua
<https://en.wikipedia.org/wiki/Quechuan_languages>`__
word *chuqllu*.

.. seealso::

    Choclo is a part of the
    `Fatiando a Terra <https://www.fatiando.org/>`__ project.

Overview
--------

Choclo provides slim and optimized function to compute the gravitational and
magnetic fields of simple geometries like point masses, magnetic dipoles and
prisms. It also provides the *kernel* functions needed to run compute those
fields. The goal of Choclo is to provide developers of a simple and efficient
way to calculate these fields for a wide range of applications, like forward
modellings, sensitivity matrices calculations, equivalent sources
implementations and more.

These functions are not designed to be used by final users. Instead they are
meant to be part of the underlaying engine of a higher level codebase, like
`Harmonica <https://www.fatiando.org/harmonica>`__.

All Choclo functions rely on `Numba <https://numba.pydata.org/>`__ for
just-in-time compilations, meaning that there's no need to distribute
precompiled code: Choclo provides pure Python code that gets compiled during
runtime allowing to run them as fast as they were written in C.
Moreover, developers could harness the power of Numba to parallelize processes
in a quick and easy way.

Conventions
-----------

Before you jump into Choclo's functions, it's worth noting some conventions
that will be kept along its codebase:

- The functions assume a right-handed coordinate system. We avoid using names
  like "x", "y" and "z" for the coordinates. Instead we use "easting",
  "northing" and "upward" to make extra clear the direction of each axis.
- We use the first letter of the *easting*, *northing* and *upward* axis to
  indicate direction of derivatives. For example, a function ``gravity_e`` will
  compute the *easting* component of the gravitional acceleration, while the
  ``gravity_n`` and ``gravity_u`` will compute the *northing* and *upward*
  ones, respectively.
- The arguments of the functions are always assumed in SI units. And all the
  functions return results also in SI units. Choclo **doesn't** perform **unit
  conversions**.
- The components of the gravitational accelerations and the magnetic fields are
  computed in the same direction of the *easting*, *northing* and *upward*
  axis. So ``gravity_u`` will compute the **upward** component of the
  gravitational acceleration (note the difference with the **downward**
  component).


The library
-----------

Choclo is divided in a few different submodules, each with different goals. The
three main modules are the ones that host the forward and kernel functions for
the different geometries supported by Choclo: ``point``, ``dipole`` and
``prism``. Inside each one of these modules we will find forward modelling
functions and potentially some kernel functions. The names of the forward
modelling functions follow a simple pattern of ``{field}_{type}``. For
example, :func:`choclo.prism.gravity_e` computes the easting component of the
gravitational acceleration of a prism, while :func:`choclo.prism.gravity_ee`
computes the easting-easting gravity tensor component.


How to use Choclo
-----------------

The simplest case
~~~~~~~~~~~~~~~~~

Using Choclo is very simple, but it requires some work from our side. Let's say
we need to compute the upward component of the gravitational acceleration that
a single rectangular prism produces on a single computation point. To do so we
can just call the :func:`choclo.prism.gravity_u` function:

.. jupyter-execute::

    import numpy as np
    from choclo.prism import gravity_u

    # Define a single computation point
    easting, northing, upward = 0.0, 0.0, 10.0

    # Define the boundaries of the prism as a 1d-array
    prism = np.array([-10.0, 10.0, -7.0, 7.0, -15.0, -5.0])

    # And its density
    density = 400.0

    # Compute the upward component of the grav. acceleration
    g_u = gravity_u(easting, northing, upward, prism, density)
    g_u

But this case is very simple: we usually deal with multiple sources and
multiple computation points.

Multiple sources and computation points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Reference Documentation

    api/index.rst
    citing.rst
    references.rst


.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Community

    Join the community <http://contact.fatiando.org>
    How to contribute <https://github.com/fatiando/choclo/blob/main/CONTRIBUTING.md>
    Code of Conduct <https://github.com/fatiando/choclo/blob/main/CODE_OF_CONDUCT.md>
    Source code on GitHub <https://github.com/fatiando/choclo>
    The Fatiando a Terra project <https://www.fatiando.org>
