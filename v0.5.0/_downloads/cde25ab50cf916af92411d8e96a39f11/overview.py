#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


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


# In[4]:


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

