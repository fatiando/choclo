.. _install:

Installing
==========

Which Python?
-------------

You'll need **Python 3.8 or greater**.
See :ref:`python-versions` if you require support for older versions.

We recommend using the
`Anaconda Python distribution <https://www.anaconda.com/download>`__
to ensure you have all dependencies installed and the ``conda`` package manager
available.
Installing Anaconda does not require administrative rights to your computer and
doesn't interfere with any other Python installations in your system.


Dependencies
------------

The required dependencies should be installed automatically when you install
Choclo using ``conda`` or ``pip``. Optional dependencies have to be
installed manually.

.. note::

    See :ref:`dependency-versions` for the our policy of oldest supported
    versions of each dependency.

Required:

* `numpy <http://www.numpy.org/>`__
* `numba <https://numba.pydata.org/>`__

The examples in the :ref:`overview` also use:

* `matplotlib <https://matplotlib.org/>`__ for plotting

Installing with conda
---------------------

You can install Choclo using the `conda package manager
<https://conda.io/>`__ that comes with the Anaconda distribution::

    conda install choclo --channel conda-forge


Installing with pip
-------------------

Alternatively, you can also use the `pip package manager
<https://pypi.org/project/pip/>`__::

    pip install choclo


Installing the latest development version
-----------------------------------------

You can use ``pip`` to install the latest source from Github::

    pip install https://github.com/fatiando/choclo/archive/main.zip

Alternatively, you can clone the git repository locally and install from
there::

    git clone https://github.com/fatiando/choclo.git
    cd choclo
    pip install .
