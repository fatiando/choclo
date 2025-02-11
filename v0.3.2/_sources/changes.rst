.. _changes:

Changelog
=========

Version 0.3.2
-------------

Released on: 2025/02/11

doi: https://doi.org/10.5281/zenodo.14852111

Bug Fixes:

-  Fix bug in third order prism kernels ``kernel_iij`` that triggered division by zero errors and numerical instabilities (`#118 <https://github.com/fatiando/choclo/pull/118>`__)

Maintenance:

-  Use ``zizmor`` to lint GitHub Actions for security vulnerabilities(`#114 <https://github.com/fatiando/choclo/pull/114>`__)
-  Install ``zizmor`` through ``conda-forge`` in ``environment.yml`` (`#115 <https://github.com/fatiando/choclo/pull/115>`__)
-  Fix ``zizmor`` error on ``excessive-permissions`` (`#119 <https://github.com/fatiando/choclo/pull/119>`__)
-  Fix deployment of docs after limiting lifetime of credentials in Action workflow (`#122 <https://github.com/fatiando/choclo/pull/122>`__)
-  Replace ``flake8``, ``isort`` and ``black`` for ``ruff`` (`#112 <https://github.com/fatiando/choclo/pull/112>`__)

This release contains contributions from:

-  Santiago Soler


Version 0.3.1
-------------

Released on: 2024/11/26

doi: https://doi.org/10.5281/zenodo.14226707

Bug fixes:

-  Make 2nd and 3rd order prism kernels return ``np.nan`` on vertices (`#108 <https://github.com/fatiando/choclo/pull/108>`__)

Documentation:

-  Group similar parameters in docstrings (`#105 <https://github.com/fatiando/choclo/pull/105>`__)

This release contains contributions from:

-  Santiago Soler

Version 0.3.0
-------------

Released on: 2024/10/08

doi: https://doi.org/10.5281/zenodo.13905447

Bug fixes:

-  Fix bug on ``safe_log`` and solves discontinuous magnetic fields (`#100 <https://github.com/fatiando/choclo/pull/100>`__)

New features:

-  Add forward modelling functions for the magnetic gradiometry components of prisms (`#97 <https://github.com/fatiando/choclo/pull/97>`__)

Maintenance:

-  Run tests with oldest dependencies on x86 macos (`#83 <https://github.com/fatiando/choclo/pull/83>`__)
-  Replace ``_version_generated.py`` for ``_version.py`` in Makefile (`#82 <https://github.com/fatiando/choclo/pull/82>`__)
-  Update how output variables are stored in Actions (`#90 <https://github.com/fatiando/choclo/pull/90>`__)
-  Move push to codecov to its own job in Actions (`#88 <https://github.com/fatiando/choclo/pull/88>`__)
-  Replace ``build`` for ``python-build`` in ``environment.yml`` (`#91 <https://github.com/fatiando/choclo/pull/91>`__)
-  Simplify tests for prism magnetic forward funcs (`#96 <https://github.com/fatiando/choclo/pull/96>`__)
-  Add some more tests for ``_safe_log`` (`#101 <https://github.com/fatiando/choclo/pull/101>`__)

Documentation:

-  Replace Sphinx napoleon for numpydoc (`#84 <https://github.com/fatiando/choclo/pull/84>`__)
-  Fix style errors in docstrings (`#85 <https://github.com/fatiando/choclo/pull/85>`__)
-  Improve docstrings of ``magnetic_field`` functions (`#87 <https://github.com/fatiando/choclo/pull/87>`__)

This release contains contributions from:

-  Santiago Soler


Version 0.2.0
-------------

Released on: 2024/04/09

doi: https://doi.org/10.5281/zenodo.10951580

New features:

* Restore kernel functions for forward modelling of point sources (`#58
  <https://github.com/fatiando/choclo/pull/58>`__)

Documentation:

* Update the versions of Sphinx and its plugins, including dark theme and minor
  changes to the style of the docs (`#64
  <https://github.com/fatiando/choclo/pull/64>`__)
* Mention SimPEG's support and collaboration in docs (`#65
  <https://github.com/fatiando/choclo/pull/65>`__)

Maintenance:

* Use pip instead of conda for the docs workflow (`#63
  <https://github.com/fatiando/choclo/pull/63>`__)
* Use Burocrata to check/add license notices (`#66
  <https://github.com/fatiando/choclo/pull/66>`__)
* Use Dependabot to manage GitHub Actions updates (`#68
  <https://github.com/fatiando/choclo/pull/68>`__)
* Setup Trusted Publisher deployment to PyPI (`#67
  <https://github.com/fatiando/choclo/pull/67>`__)
* Ditch setup.cfg and replace it with pyproject.toml (`#77
  <https://github.com/fatiando/choclo/pull/77>`__)
* Replace `_version_generated.py` for `_version.py` (`#80
  <https://github.com/fatiando/choclo/pull/80>`__)
* Extend support for Python 3.12 (`#79
  <https://github.com/fatiando/choclo/pull/79>`__)
* Drop support for Python 3.7 (`#78
  <https://github.com/fatiando/choclo/pull/78>`__)
* Update Black formatting to version 24.2 (`#61
  <https://github.com/fatiando/choclo/pull/61>`__)

This release contains contributions from:

* Santiago Soler
* Leonardo Uieda


Version 0.1.0
-------------

Released on: 2023/05/12

doi: https://doi.org/10.5281/zenodo.7931023

Breaking changes:

* Ditch kernel functions for point masses (`#42 <https://github.com/fatiando/choclo/pull/42>`__)
* Make Choclo functions to take only scalar inputs instead of array inputs (`#50 <https://github.com/fatiando/choclo/pull/50>`__)
* Update the value of the gravitational constant to the latest standard (`#56 <https://github.com/fatiando/choclo/pull/56>`__)

Bug fixes:

* Fix bug on non-diagonal tensor components of prisms (`#27 <https://github.com/fatiando/choclo/pull/27>`__)

New features:

* Gravity and magnetic forward models for prisms return nan on singular points (`#30 <https://github.com/fatiando/choclo/pull/30>`__)

Maintenance:

* Drop support for Python 3.6 (`#29 <https://github.com/fatiando/choclo/pull/29>`__)
* Add missing matplotlib to the dev environment (`#44 <https://github.com/fatiando/choclo/pull/44>`__)

Documentation:

* Add installation instructions to the docs (`#35 <https://github.com/fatiando/choclo/pull/35>`__)
* Improve Overview page in docs: avoid printing huge arrays, and add plots (`#37 <https://github.com/fatiando/choclo/pull/37>`__)
* Add buttons to download user guide pages (`#39 <https://github.com/fatiando/choclo/pull/39>`__)
* Add User Guide with example for building jacobians (`#40 <https://github.com/fatiando/choclo/pull/40>`__)
* Move "How to use Choclo" to its own user guide page (`#41 <https://github.com/fatiando/choclo/pull/41>`__)
* Improve math in Jacobian matrix user guide page (`#43 <https://github.com/fatiando/choclo/pull/43>`__)
* Add Zenodo doi for all versions for citation (`#45 <https://github.com/fatiando/choclo/pull/45>`__)
* Add changelog and links to docs for other versions (`#46 <https://github.com/fatiando/choclo/pull/46>`__)
* Add a logo for Choclo, based on colorful corn variations from the Andes (`#48 <https://github.com/fatiando/choclo/pull/48>`__)

This release contains contributions from:

* Santiago Soler
* Leonardo Uieda


Version 0.0.1
-------------

Released on: 2022/11/19

doi: https://doi.org/10.5281/zenodo.7851748

First release of Choclo, including functions for gravity and magnetic forward
modelling of point sources and right-rectangular prisms. This first release
will serve as a test of the API concept as we trial the use Choclo in other
projects.

This release contains contributions from:

* Santiago Soler
* Leonardo Uieda
