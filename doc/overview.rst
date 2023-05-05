.. _overview:

Overview
========

Choclo provides slim and optimized functions to compute the gravitational and
magnetic fields of simple geometries like point masses, magnetic dipoles and
prisms. It also provides the *kernel* functions needed to run compute those
fields. The goal of Choclo is to provide developers a simple and efficient
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
  compute the *easting* component of the gravitational acceleration, while the
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

Gravity forward modelling
~~~~~~~~~~~~~~~~~~~~~~~~~

Choclo offers functions to forward model the gravity fields of point masses and
rectangular prisms.

For example, we can compute the gravity acceleration and gravity tensor
components of a single point mass on a single observation point:

.. jupyter-execute::

   import choclo

   # Define a single point mass located 10 meters below the zero height
   easting_m, northing_m, upward_m = 0., 0., -10.
   mass = 1e4  # mass of the source in kg

   # Define coordinates of an observation point 2 meters above the zero height
   easting, northing, upward = 0., 0., 2.

   # Compute the upward compont of the gravity acceleration the point mass
   # generates on this observation point (in SI units)
   g_u = choclo.point.gravity_u(
      easting, northing, upward, easting_m, northing_m, upward_m, mass
   )
   print(f"g_u: {g_u:.2e} m s^(-2)")

   # Compute gravity tensor components (in SI units)
   g_eu = choclo.point.gravity_eu(
      easting, northing, upward, easting_m, northing_m, upward_m, mass
   )
   g_uu = choclo.point.gravity_uu(
      easting, northing, upward, easting_m, northing_m, upward_m, mass
   )
   print(f"g_eu: {g_eu:.2e} s^(-2)")
   print(f"g_uu: {g_uu:.2e} s^(-2)")


We can do something similar for a rectangular prism:

.. jupyter-execute::

   import numpy as np

   # Define a single rectangular prism through its boundaries
   west, east, south, north, bottom, top = -1., 5., -4., 4., -20., -10.
   density = 2900  # density of the prism in kg m^(-3)

   # Define coordinates of an observation point
   easting, northing, upward = 1., 3., -1.

   # Compute the upward compont of the gravity acceleration the prism
   # generates on this observation point (in SI units)
   g_u = choclo.prism.gravity_u(
       easting, northing, upward, west, east, south, north, bottom, top, density,
   )
   print(f"g_u: {g_u:.2e} m s^(-2)")

   # Compute gravity tensor components (in SI units)
   g_nu = choclo.prism.gravity_nu(
       easting, northing, upward, west, east, south, north, bottom, top, density,
   )
   g_ee = choclo.prism.gravity_ee(
       easting, northing, upward, west, east, south, north, bottom, top, density,
   )
   print(f"g_nu: {g_nu:.2e} s^(-2)")
   print(f"g_ee: {g_ee:.2e} s^(-2)")


Magnetic forward modelling
~~~~~~~~~~~~~~~~~~~~~~~~~~

Choclo also offers functions for computing the magnetic field of dipoles and
rectangular prisms. We can choose to compute the three components at once (using
functions like :func:`choclo.dipole.magnetic_field` and
:func:`choclo.prism.magnetic_field`), or one component at a time (see
:func:`choclo.dipole.magnetic_e` and :func:`choclo.prism.magnetic_u` for
example).

For example, we can compute the three magnetic field components of a dipole on
a single observation point:

.. jupyter-execute::

   # Define the location of a dipole
   easting_d, northing_d, upward_d = -4., 2., -1.

   # Define the magnetic moment vector of the dipole (in A m^2)
   mag_moment_e, mag_moment_n, mag_moment_u = 1., 1., -2.

   # Define coordinates of an observation point
   easting, northing, upward = -2., 2., 2.

   # Compute the magnetic field of the dipole on the observation point (in T)
   b_e, b_n, b_u = choclo.dipole.magnetic_field(
      easting,
      northing,
      upward,
      easting_d,
      northing_d,
      upward_d,
      mag_moment_e,
      mag_moment_n,
      mag_moment_u,
   )
   print(f"b_e: {b_e:.2e} T")
   print(f"b_n: {b_n:.2e} T")
   print(f"b_u: {b_u:.2e} T")


Or the upward component of the magnetic field generated by a prism:

.. jupyter-execute::

   # Define a rectangular prism
   west, east, south, north, bottom, top = -1., 5., -4., 4., -20., -10.

   # Define its magnetization vector (in A m^(-1))
   m_e, m_n, m_u = 0.5, -1.5, -1.3

   # Define coordinates of an observation point
   easting, northing, upward = 3., 0., -1.

   # Compute the upward component of the magnetic field of the prism (in T)
   b_u = choclo.prism.magnetic_u(
       easting, northing, upward, west, east, south, north, bottom, top, m_e, m_n, m_u,
   )
   print(f"b_u: {b_u:.2e} T")

.. important::

   Computing the three components independently is less efficient than
   computing them all at once using the :func:`choclo.dipole.magnetic_field` or
   :func:`choclo.prism.magnetic_field` functions.

.. seealso::

   :ref:`howtouse` provides detailed instructions on how to use Choclo to
   efficiently compute gravity and magnetic fields of multiple sources on
   multiple observation points.

----

.. grid:: 2

    .. grid-item-card:: :jupyter-download-script:`Download Python script <overview>`
        :text-align: center

    .. grid-item-card:: :jupyter-download-nb:`Download Jupyter notebook <overview>`
        :text-align: center

