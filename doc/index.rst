.. title:: Home

.. grid::
    :gutter: 2 3 3 3
    :margin: 5 5 0 0
    :padding: 0 0 0 0

    .. grid-item::
        :columns: 12 8 8 8

        .. raw:: html

            <h1 class="display-1">Choclo</h1>

        .. div:: sd-fs-3

            Kernel functions for your geophysical models

    .. grid-item::
        :columns: 12 4 4 4

        .. image:: ./_static/choclo-logo.svg
            :width: 200px
            :class: sd-m-auto dark-light

**Choclo** is a Python library that hosts optimized forward modelling and
kernel functions for running geophysical forward and inverse models, intended
to be used by other libraries as the underlying layer of their computation.

Choclo is part of the `Fatiando a Terra <https://www.fatiando.org/>`__ project,
and built in collaboration with `SimPEG <https://simpeg.xyz>`__ with the goal
of developing a common engine for running gravity and magnetic forward models.

.. hint::

   "Choclo" is a term used in some countries of South America to refer to corn,
   originated from the `quechua
   <https://en.wikipedia.org/wiki/Quechuan_languages>`__
   word *chuqllu*.


.. grid:: 1 2 1 2
    :margin: 5 5 0 0
    :padding: 0 0 0 0
    :gutter: 4

    .. grid-item-card:: :octicon:`rocket` Getting started
        :text-align: center
        :class-title: sd-fs-5
        :class-card: sd-p-3

        New to Choclo? Start here!

        .. button-ref:: overview
            :click-parent:
            :color: primary
            :outline:
            :expand:

    .. grid-item-card:: :octicon:`comment-discussion` Need help?
        :text-align: center
        :class-title: sd-fs-5
        :class-card: sd-p-3

        Ask on our community channels

        .. button-link:: https://www.fatiando.org/contact
            :click-parent:
            :color: primary
            :outline:
            :expand:

             Join the conversation

    .. grid-item-card:: :octicon:`file-badge` Reference documentation
        :text-align: center
        :class-title: sd-fs-5
        :class-card: sd-p-3

        A list of modules and functions

        .. button-ref:: api
            :click-parent:
            :color: primary
            :outline:
            :expand:

    .. grid-item-card:: :octicon:`bookmark` Using Choclo for research?
        :text-align: center
        :class-title: sd-fs-5
        :class-card: sd-p-3

        Citations help support our work

        .. button-ref:: citing
            :click-parent:
            :color: primary
            :outline:
            :expand:


.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Getting Started

    overview.rst
    install.rst

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: User Guide

    user_guide/how-to-use.rst
    user_guide/jacobian.rst

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Reference Documentation

    api/index.rst
    citing.rst
    references.rst
    changes.rst
    compatibility.rst
    versions.rst


.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Community

    Join the community <http://contact.fatiando.org>
    How to contribute <https://github.com/fatiando/choclo/blob/main/CONTRIBUTING.md>
    Code of Conduct <https://github.com/fatiando/choclo/blob/main/CODE_OF_CONDUCT.md>
    Source code on GitHub <https://github.com/fatiando/choclo>
    The Fatiando a Terra project <https://www.fatiando.org>
    SimPEG <https://simpeg.xyz>
