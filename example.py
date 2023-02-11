import numba
import numpy as np
import matplotlib.pyplot as plt

from choclo.prism import gravity_u

prisms = np.array(
    [
        [-10.0, 0.0, -7.0, 0.0, -15.0, -10.0],
        [-10.0, 0.0, 0.0, 7.0, -25.0, -15.0],
        [0.0, 10.0, -7.0, 0.0, -20.0, -13.0],
        [0.0, 10.0, 0.0, 7.0, -12.0, -8.0],
    ]
)
densities = np.array([200.0, 300.0, -100.0, 400.0])

easting = np.linspace(-5.0, 5.0, 21)
northing = np.linspace(-4.0, 4.0, 21)
easting, northing = np.meshgrid(easting, northing)
upward = 10 * np.ones_like(easting)
coordinates = (easting, northing, upward)


def gravity_upward_slow(coordinates, prisms, densities):
    """
    Compute the upward component of the acceleration of a set of prisms
    """
    # Keep record of the shape of the coordinates arrays
    cast = np.broadcast(*coordinates)
    # Unpack coordinates of the observation points and ravel them
    easting, northing, upward = tuple(c.ravel() for c in coordinates[:])
    # Initialize a result array full of zeros
    result = np.zeros(cast.size, dtype=np.float64)
    # Compute the upward component that every prism generate on each
    # observation point
    for i in range(len(easting)):
        for j in range(prisms.shape[0]):
            result[i] += gravity_u(
                easting[i], northing[i], upward[i], prisms[j, :], densities[j]
            )
    return result.reshape(cast.shape)


g_u = gravity_upward_slow(coordinates, prisms, densities)

# Plot the resulting field
plt.pcolormesh(easting, northing, g_u, shading="auto")
plt.gca().set_aspect("equal")
plt.colorbar()
plt.show()


@numba.jit(nopython=True)
def gravity_upward_jit(coordinates, prisms, densities):
    """
    Compute the upward component of the acceleration of a set of prisms
    """
    # Keep record of the shape of the coordinates arrays
    cast = np.broadcast(*coordinates)
    # Unpack coordinates of the observation points and ravel them
    easting, northing, upward = tuple(c.ravel() for c in coordinates[:])
    # Initialize a result array full of zeros
    result = np.zeros(cast.size, dtype=np.float64)
    # Compute the upward component that every prism generate on each
    # observation point
    for i in range(len(easting)):
        for j in range(prisms.shape[0]):
            result[i] += gravity_u(
                easting[i], northing[i], upward[i], prisms[j, :], densities[j]
            )
    return result.reshape(cast.shape)


g_u = gravity_upward_jit(coordinates, prisms, densities)

# Plot the resulting field
plt.pcolormesh(easting, northing, g_u, shading="auto")
plt.gca().set_aspect("equal")
plt.colorbar()
plt.show()


@numba.jit(nopython=True, parallel=True)
def gravity_upward_parallel(coordinates, prisms, densities):
    """
    Compute the upward component of the acceleration of a set of prisms
    """
    # Keep record of the shape of the coordinates arrays
    cast = np.broadcast(*coordinates)
    # Unpack coordinates of the observation points and ravel them
    easting, northing, upward = tuple(c.ravel() for c in coordinates[:])
    # Initialize a result array full of zeros
    result = np.zeros(cast.size, dtype=np.float64)
    # Compute the upward component that every prism generate on each
    # observation point
    for i in numba.prange(len(easting)):
        for j in range(prisms.shape[0]):
            result[i] += gravity_u(
                easting[i], northing[i], upward[i], prisms[j, :], densities[j]
            )
    return result.reshape(cast.shape)


g_u = gravity_upward_parallel(coordinates, prisms, densities)

# Plot the resulting field
plt.pcolormesh(easting, northing, g_u, shading="auto")
plt.gca().set_aspect("equal")
plt.colorbar()
plt.show()
