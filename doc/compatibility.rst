.. _compatibility:

Version compatibility
=====================

Choclo version compatibility
-------------------------------

Choclo uses `semantic versioning <https://semver.org/>`__ (i.e.,
``MAJOR.MINOR.BUGFIX`` format).

* Major releases mean that backwards incompatible changes were made.
  Upgrading will require users to change their code.
* Minor releases add new features/data without changing existing functionality.
  Users can upgrade minor versions without changing their code.
* Bug fix releases fix errors in a previous release without adding new
  functionality. Users can upgrade minor versions without changing their code.

We will add ``FutureWarning`` messages about deprecations ahead of making any
breaking changes to give users a chance to upgrade.

.. warning::

    The above does not apply to versions < ``1.0.0``. All ``0.*`` versions may
    deprecate, remove, or change functionality between releases. Proper
    warnings will be raised and any breaking changes will be marked as such in
    the :ref:`changes`.

.. _dependency-versions:

Supported dependency versions
-----------------------------

Choclo follows the recommendations in
`NEP29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`__ for setting
the minimum required version of our dependencies.
In short, we support **all minor releases of our dependencies from the previous
24 months** before a Choclo release with a minimum of 2 minor releases.

We follow this guidance conservatively and won't require newer versions if the
older ones are still working without causing problems.
Whenever support for a version is dropped, we will include a note in the
:ref:`changes`.


.. _python-versions:

Supported Python versions
-------------------------

If you require support for older Python versions, please pin Choclo to the
following releases to ensure compatibility:

.. list-table::
    :widths: 40 60

    * - **Python version**
      - **Last compatible release**
    * - 3.7
      - 0.1.0
    * - 3.8
      - 0.3.2
