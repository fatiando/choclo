.. _changes:

Changelog
=========

Version 0.1.0
-------------

Released on: 2023/05/12

doi: https://doi.org/10.5281/zenodo.7931023

Breaking changes:

- Ditch kernel functions for point masses (`#42 <https://github.com/fatiando/choclo/pull/42>`__)
- Make Choclo functions to take only scalar inputs instead of array inputs  (`#50 <https://github.com/fatiando/choclo/pull/50>`__)

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
