Building a Jacobian matrix
--------------------------

In several applications, like 3D inversions, we need to build the *Jacobian
matrix*, a.k.a *sensitivity matrix* of our forward model for every observation
point, i.e. how much does our field changes when I apply an infinitesimal
change to the physical property of each source in the model.

Let's take for example the upward component of the gravity acceleration of
prisms. Given :math:`N` observation points and :math:`M` prisms, the gravity
field on the :math:`i`-th observation point :math:`\mathbf{p}_i` can be
computed as:

.. math::

   g_u(\mathbf{p}_i) = \sum\limits_{j=1}^M u_j(\mathbf{p}) \rho_j

where :math:`\rho_j` is the density of the :math:`j`-th prism and
:math:`u_j(\mathbf{p})` represents the forward modelling function for the
:math:`j`-th rectangular prism on the observation point :math:`\mathbf{p}`.

The Jacobian matrix :math:\mathbf{J}` is an :math:`N \times M` matrix whose
elements are the partial derivative of :math:`g_u(\mathbf{p_i})` with respect to
the density of the :math:`j`-th prism:

.. math::

   J_{ij} = \frac{\partial g_u(\mathbf{p}_i)}{\partial \rho_j} = u_j(\mathbf{p}_i)

In most potential field cases, the forward model is linear on the physical
property (density in this case), so the Jacobian elements are constant.

So, in order to build the sensitivity matrix we must need to evaluate every
:math:`u_j` on every observation point :math:`\mathbf{p}`. We can easily do so
with Choclo forward modelling functions, considering that the source has unit
density.

Let's build a function that can build a sensitivity matrix for a set of
observation points and a set of prisms. Since this operation is as demanding as
forward modelling our entire set of prisms on every observation point, we
want it to run fast and optionally in parallel. Therefore, we are going to
write a Numba function with parallelization enabled:

.. jupyter-execute::

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
                   easting[i], northing[i], upward[i], prisms[j, :], 1.0
               )
       return jacobian

.. note::

   The :func:`numpy.empty` function creates an *empty* array. This means it
   allocates the memory for this array, but it doesn't write any values in it.
   Instead, the ``jacobian`` array is filled with garbage values after being
   initialized.

   By using :func:`numpy.empty` we are saving some time, avoiding to fill the
   ``jacobian`` array with values that we will soon overwrite in the following
   for loops.

Let's try this function by defining some prisms and observation points:

.. jupyter-execute::

   easting = np.linspace(-5.0, 5.0, 21)
   northing = np.linspace(-4.0, 4.0, 21)
   easting, northing = np.meshgrid(easting, northing)
   upward = 10 * np.ones_like(easting)

   coordinates = (easting.ravel(), northing.ravel(), upward.ravel())


.. jupyter-execute::

   prisms = np.array(
       [
           [-10.0, 0.0, -7.0, 0.0, -15.0, -10.0],
           [-10.0, 0.0, 0.0, 7.0, -25.0, -15.0],
           [0.0, 10.0, -7.0, 0.0, -20.0, -13.0],
           [0.0, 10.0, 0.0, 7.0, -12.0, -8.0],
       ]
   )

And run it:

.. jupyter-execute::

   jacobian = build_jacobian(coordinates, prisms)
   jacobian

.. warning::

   Jacobian matrices can be very big. Large number of observation points and
   sources can lead to Jacobian matrices that cannot fit in the available
   memory of your system.

Now that we have defined our Jacobian matrix, we can use it to forward model
the gravity field of our prisms on every observation point by just computing
a dot product between it and the density vector of the prisms
(:math:`\mathbf{m}`):

.. math::

   \mathbf{g_u}
   =
   \begin{bmatrix}
   g_u({\mathbf{p}_1}) \\
   \vdots \\
   g_u({\mathbf{p}_N}) \\
   \end{bmatrix}
   =
   \begin{bmatrix}
   J_{11} & \cdots & J_{1M} \\
   \vdots & \ddots & \vdots \\
   J_{N1} & \cdots & J_{NM}
   \end{bmatrix}
   \cdot
   \begin{bmatrix}
   \rho_1 \\
   \vdots \\
   \rho_M \\
   \end{bmatrix}
   =
   \mathbf{J} \cdot \mathbf{m}

.. jupyter-execute::

   # Define densities for the prisms
   densities = np.array([200.0, 300.0, -100.0, 400.0])

   # Compute result
   g_u = jacobian @ densities

We can check that this result is right by comparing it with the output of the
``gravity_u_parallel`` function we defined in the :ref:`overview`:

.. jupyter-execute::
   :hide-code:

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
                   easting[i], northing[i], upward[i], prisms[j, :], densities[j]
               )
       return result

.. jupyter-execute::

   expected = gravity_upward_parallel(coordinates, prisms, densities)
   np.allclose(g_u, expected)
