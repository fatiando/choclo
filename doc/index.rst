.. title:: Home

Choclo
======

.. div:: sd-fs-3

    Kernel functions for your geophysical models

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

----


Project status
--------------

**Choclo is ready for use but still changing.** This means that we
sometimes break backwards compatibility as we try to improve the
software based on user experience, new ideas, better design decisions,
etc. Please keep that in mind before you update Choclo to a newer
version.

**We welcome feedback and ideas!** This is a great time to bring new
ideas on how we can improve the project. `Join the
conversation <https://www.fatiando.org/contact>`__ or submit `issues on
GitHub <https://github.com/fatiando/choclo/issues>`__.

Getting involved
----------------

🗨️ **Contact us:** Find out more about how to reach us at
`fatiando.org/contact <https://www.fatiando.org/contact/>`__.

👩🏾‍💻 **Contributing to project development:** Please read our
`Contributing
Guide <https://github.com/fatiando/choclo/blob/main/CONTRIBUTING.md>`__
to see how you can help and give feedback.

🤖 **Policy on AI usage:** Please read our `AI Usage
Policy <https://github.com/fatiando/community/blob/main/AI.md>`__.

🧑🏾‍🤝‍🧑🏼 **Code of conduct:** This project is released with a `Code of
Conduct <https://github.com/fatiando/community/blob/main/CODE_OF_CONDUCT.md>`__.
By participating in this project you agree to abide by its terms.

   **Imposter syndrome disclaimer:** We want your help. **No, really.**
   There may be a little voice inside your head that is telling you that
   you’re not ready, that you aren’t skilled enough to contribute. We
   assure you that the little voice in your head is wrong. Most
   importantly, **there are many valuable ways to contribute besides
   writing code**.

   *This disclaimer was adapted from the* `MetPy
   project <https://github.com/Unidata/MetPy>`__.

License
-------

This is free software: you can redistribute it and/or modify it under
the terms of the **BSD 3-clause License**. A copy of this license is
provided in
`LICENSE.txt <https://github.com/fatiando/choclo/blob/main/LICENSE.txt>`__.


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
    user_guide/legendre.rst

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

    Join our community <https://www.fatiando.org/contact>
    How to contribute <https://github.com/fatiando/choclo/blob/main/CONTRIBUTING.md>
    Code of Conduct <https://github.com/fatiando/choclo/blob/main/CODE_OF_CONDUCT.md>
    Authors <https://github.com/fatiando/choclo/blob/main/AUTHORS.md>
    The Fatiando a Terra project <https://www.fatiando.org>
    SimPEG <https://simpeg.xyz>

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Code and packages

    GitHub <https://github.com/fatiando/choclo>
    PyPI <https://pypi.python.org/pypi/choclo>
    conda-forge <https://github.com/conda-forge/choclo-feedstock>
    Zenodo <https://doi.org/10.5281/zenodo.7851747>
