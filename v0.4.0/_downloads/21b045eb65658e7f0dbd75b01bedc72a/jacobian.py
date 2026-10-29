#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numba
import numpy as np
from choclo.prism import gravity_u

@numba.jit(nopython=True, parallel=True)
def build_jacobian(coordinates, prisms):
    """
    Build a sensitivity matrix for gravity_u of a prism
    """
    # Unpack coordinates of the observation points
    easting, northing, upward = coordinates[:]
    # Initialize an empty 2d array for the sensitivity matrix
    n_coords = easting.size
    n_prisms = prisms.shape[0]
    jacobian = np.empty((n_coords, n_prisms), dtype=np.float64)
    # Compute the gravity_u field that each prism generate on every observation
    # point, considering that they have a unit density
    for i in numba.prange(len(easting)):
        for j in range(prisms.shape[0]):
            jacobian[i, j] = gravity_u(
                easting[i],
                northing[i],
                upward[i],
                prisms[j, 0],
                prisms[j, 1],
                prisms[j, 2],
                prisms[j, 3],
                prisms[j, 4],
                prisms[j, 5],
                1.0,
            )
    return jacobian


# In[2]:


easting = np.linspace(-5.0, 5.0, 21)
northing = np.linspace(-4.0, 4.0, 21)
easting, northing = np.meshgrid(easting, northing)
upward = 10 * np.ones_like(easting)

coordinates = (easting.ravel(), northing.ravel(), upward.ravel())


# In[3]:


prisms = np.array(
    [
        [-10.0, 0.0, -7.0, 0.0, -15.0, -10.0],
        [-10.0, 0.0, 0.0, 7.0, -25.0, -15.0],
        [0.0, 10.0, -7.0, 0.0, -20.0, -13.0],
        [0.0, 10.0, 0.0, 7.0, -12.0, -8.0],
    ]
)


# In[4]:


jacobian = build_jacobian(coordinates, prisms)
jacobian


# In[5]:


# Define densities for the prisms
densities = np.array([200.0, 300.0, -100.0, 400.0])

# Compute result
g_u = jacobian @ densities


# In[6]:


@numba.jit(nopython=True, parallel=True)
def gravity_upward_parallel(coordinates, prisms, densities):
    """
    Compute the upward component of the acceleration of a set of prisms
    """
    # Unpack coordinates of the observation points
    easting, northing, upward = coordinates[:]
    # Initialize a result array full of zeros
    result = np.zeros_like(easting, dtype=np.float64)
    # Compute the upward component that every prism generate on each
    # observation point
    for i in numba.prange(len(easting)):
        for j in range(prisms.shape[0]):
            result[i] += gravity_u(
                easting[i],
                northing[i],
                upward[i],
                prisms[j, 0],
                prisms[j, 1],
                prisms[j, 2],
                prisms[j, 3],
                prisms[j, 4],
                prisms[j, 5],
                densities[j],
            )
    return result


# In[7]:


expected = gravity_upward_parallel(coordinates, prisms, densities)
np.allclose(g_u, expected)

