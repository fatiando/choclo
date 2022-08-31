.. _api:

List of functions and classes (API)
===================================

.. automodule:: choclo

.. currentmodule:: choclo

Kernel functions
----------------

Here you will find the list of available kernel functions for gravity and
magnetic forward modellings.

Point sources
~~~~~~~~~~~~~

For point sources and observation points defined in **Cartesian coordinates**.

Potential field
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

    kernel_point_potential

Gradient components
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

    kernel_point_g_easting
    kernel_point_g_northing
    kernel_point_g_upward

Tensor components
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

    kernel_point_g_ee
    kernel_point_g_nn
    kernel_point_g_zz
    kernel_point_g_en
    kernel_point_g_ez
    kernel_point_g_nz


Euclidean distances
-------------------

Use these functions to compute Euclidean distance in Cartesian and spherical
coordinates:

.. autosummary::
   :toctree: generated/

    distance_cartesian
    distance_spherical
    distance_spherical_core

