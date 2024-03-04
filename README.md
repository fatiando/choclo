<img src="https://github.com/fatiando/choclo/raw/main/doc/_static/readme-banner.png" alt="Choclo">

<h2 align="center">Kernel functions for your geophysical models</h2>

<p align="center">
<a href="https://www.fatiando.org/choclo"><strong>Documentation</strong> (latest)</a> •
<a href="https://www.fatiando.org/choclo/dev"><strong>Documentation</strong> (main branch)</a> •
<a href="https://github.com/fatiando/choclo/blob/main/CONTRIBUTING.md"><strong>Contributing</strong></a> •
<a href="https://www.fatiando.org/contact/"><strong>Contact</strong></a>
</p>

<p align="center">
Part of the <a href="https://www.fatiando.org"><strong>Fatiando
a Terra</strong></a> project, built in collaboration with <a
href="https://simpeg.xyz"><strong>SimPEG</strong></a>
</p>

<p align="center">
<a href="https://pypi.python.org/pypi/choclo"><img src="http://img.shields.io/pypi/v/choclo.svg?style=flat-square" alt="Latest version on PyPI"></a>
<a href="https://github.com/conda-forge/choclo-feedstock"><img src="https://img.shields.io/conda/vn/conda-forge/choclo.svg?style=flat-square" alt="Latest version on conda-forge"></a>
<a href="https://codecov.io/gh/fatiando/choclo"><img src="https://img.shields.io/codecov/c/github/fatiando/choclo/main.svg?style=flat-square" alt="Test coverage status"></a>
<a href="https://pypi.python.org/pypi/choclo"><img src="https://img.shields.io/pypi/pyversions/choclo.svg?style=flat-square" alt="Compatible Python versions."></a>
<a href="https://doi.org/10.5281/zenodo.7851747"><img src="https://img.shields.io/badge/doi-10.5281%2Fzenodo.7851747-blue?style=flat-square" alt="DOI used to cite Choclo"></a>
</p>

## About

**Choclo** is a Python library that hosts optimized kernel functions for
running geophysical forward and inverse models, intended to be used by other
libraries as the underlying layer of their computation.
"Choclo" is a term used in some countries of South America to refer to corn,
originated from the [quechua](https://en.wikipedia.org/wiki/Quechuan_languages)
word _chuqllu_.

## Project goals

* Provide optimized kernel functions for gravity and magnetic forward and
  inverse models that can be easily harnessed by different geophysical
  libraries.
* Generate a pure-Python library that could be easily distributed and installed
  on any operating system.
* Rely on [Numba](https://numba.pydata.org/) for just-in-time compilations and
  optional parallelization.
* Keep the dependencies to the bare minimum to encourage other libraries to
  depend on Choclo.

## Project status

**Choclo is ready for use but still changing.**
This means that we sometimes break backwards compatibility as we try to
improve the software based on user experience, new ideas, better design
decisions, etc. Please keep that in mind before you update Choclo to a newer
version.

**We welcome feedback and ideas!** This is a great time to bring new ideas on
how we can improve the project.
[Join the conversation](https://www.fatiando.org/contact) or submit
[issues on GitHub](https://github.com/fatiando/choclo/issues).

## Getting involved

🗨️ **Contact us:**
Find out more about how to reach us at
[fatiando.org/contact](https://www.fatiando.org/contact/).

👩🏾‍💻 **Contributing to project development:**
Please read our
[Contributing Guide](https://github.com/fatiando/choclo/blob/main/CONTRIBUTING.md)
to see how you can help and give feedback.

🧑🏾‍🤝‍🧑🏼 **Code of conduct:**
This project is released with a
[Code of Conduct](https://github.com/fatiando/community/blob/main/CODE_OF_CONDUCT.md).
By participating in this project you agree to abide by its terms.

> **Imposter syndrome disclaimer:**
> We want your help. **No, really.** There may be a little voice inside your
> head that is telling you that you're not ready, that you aren't skilled
> enough to contribute. We assure you that the little voice in your head is
> wrong. Most importantly, **there are many valuable ways to contribute besides
> writing code**.
>
> *This disclaimer was adapted from the*
> [MetPy project](https://github.com/Unidata/MetPy).

## License

This is free software: you can redistribute it and/or modify it under the terms
of the **BSD 3-clause License**. A copy of this license is provided in
[`LICENSE.txt`](https://github.com/fatiando/choclo/blob/main/LICENSE.txt).
