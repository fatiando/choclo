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

For point sources and observation points defined in **Cartesian coordinates**:

.. autosummary::
   :toctree: generated/

    kernel_point_potential
    kernel_point_g_easting
    kernel_point_g_northing
    kernel_point_g_upward


Euclidean distances
-------------------

Use these functions to compute Euclidean distance in Cartesian and spherical
coordinates:

.. autosummary::
   :toctree: generated/

    distance_cartesian
    distance_spherical
    distance_spherical_core

