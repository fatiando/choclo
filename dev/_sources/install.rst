.. _install:

Installing
==========

There are different ways to install Choclo:

.. tab-set::

    .. tab-item:: conda/mamba

        Using the `conda <https://conda.io/>`__ package manager (or ``mamba``)
        that comes with the Anaconda, Miniconda, or Miniforge distributions:

        .. code:: bash

            conda install choclo --channel conda-forge

    .. tab-item:: pip

        Using the `pip <https://pypi.org/project/pip/>`__ package manager:

        .. code:: bash

            python -m pip install choclo

    .. tab-item:: Development version

        You can use ``pip`` to install the latest **unreleased** version from
        GitHub (**not recommended** in most situations):

        .. code:: bash

            python -m pip install --upgrade git+https://github.com/fatiando/choclo

.. tip::

    The commands above should be executed in a terminal. On Windows, use the
    ``cmd.exe`` or the "Anaconda Prompt" / "Miniforge Prompt" app if you're using
    Anaconda / Miniforge.

.. admonition:: Which Python?
    :class: tip

    See :ref:`python-versions` for a list of supported Python versions.

.. note::

   We recommend using the
   `Miniforge distribution <https://conda-forge.org/download/>`__
   to ensure that you have the ``conda`` package manager available.
   Installing Miniforge does not require administrative rights to your computer
   and doesn't interfere with any other Python installations in your system.
   It's also much smaller than the Anaconda distribution and is less likely to
   break when installing new software.


.. _dependencies:

Dependencies
------------

The required dependencies should be installed automatically when you install
Choclo using ``conda`` or ``pip``. Optional dependencies have to be
installed manually.

Required:

* `numpy <http://www.numpy.org/>`__
* `numba <https://numba.pydata.org/>`__

.. note::

    See :ref:`dependency-versions` for our policy of oldest supported
    versions of each dependency.

The examples in the :ref:`overview` also use:

* `matplotlib <https://matplotlib.org/>`__ for plotting
