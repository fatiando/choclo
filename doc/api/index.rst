.. _api:

List of functions and classes (API)
===================================

.. automodule:: choclo

.. currentmodule:: choclo

Point sources and dipoles
~~~~~~~~~~~~~~~~~~~~~~~~~

Forward modeling and kernel functions for point sources and dipoles in
**Cartesian coordinates**.

Gravity
^^^^^^^

.. autosummary::
   :toctree: generated/

    point.gravity_pot
    point.gravity_e
    point.gravity_n
    point.gravity_u
    point.gravity_ee
    point.gravity_nn
    point.gravity_uu
    point.gravity_en
    point.gravity_eu
    point.gravity_nu

Magnetic
^^^^^^^^

.. autosummary::
   :toctree: generated/

    dipole.magnetic_e
    dipole.magnetic_n
    dipole.magnetic_u
    dipole.magnetic_field

Kernels
^^^^^^^

.. autosummary::
   :toctree: generated/

    point.kernel_pot
    point.kernel_e
    point.kernel_n
    point.kernel_u
    point.kernel_ee
    point.kernel_nn
    point.kernel_uu
    point.kernel_en
    point.kernel_eu
    point.kernel_nu


Rectangular Prisms
~~~~~~~~~~~~~~~~~~

Forward modeling and kernel functions for right-rectangular prisms in
**Cartesian coordinates**.

Gravity
^^^^^^^

.. autosummary::
   :toctree: generated/

    prism.gravity_pot
    prism.gravity_e
    prism.gravity_n
    prism.gravity_u
    prism.gravity_ee
    prism.gravity_nn
    prism.gravity_uu
    prism.gravity_en
    prism.gravity_eu
    prism.gravity_nu

Magnetic
^^^^^^^^

.. autosummary::
   :toctree: generated/

    prism.magnetic_field
    prism.magnetic_e
    prism.magnetic_n
    prism.magnetic_u
    prism.magnetic_ee
    prism.magnetic_nn
    prism.magnetic_uu
    prism.magnetic_en
    prism.magnetic_eu
    prism.magnetic_nu

Kernels
^^^^^^^

.. autosummary::
   :toctree: generated/

    prism.kernel_pot
    prism.kernel_e
    prism.kernel_n
    prism.kernel_u
    prism.kernel_ee
    prism.kernel_nn
    prism.kernel_uu
    prism.kernel_en
    prism.kernel_eu
    prism.kernel_nu
    prism.kernel_eee
    prism.kernel_nnn
    prism.kernel_uuu
    prism.kernel_een
    prism.kernel_eeu
    prism.kernel_enn
    prism.kernel_nnu
    prism.kernel_euu
    prism.kernel_nuu
    prism.kernel_enu


Euclidean distances
~~~~~~~~~~~~~~~~~~~

Functions to compute Euclidean distance in Cartesian and spherical
coordinates:

.. autosummary::
   :toctree: generated/

    utils.distance_cartesian
    utils.distance_spherical
    utils.distance_spherical_core


Universal Constants
~~~~~~~~~~~~~~~~~~~

Universal physical constants given in SI units:

.. autosummary::
   :toctree: generated/

    constants.GRAVITATIONAL_CONST
    constants.VACUUM_MAGNETIC_PERMEABILITY
