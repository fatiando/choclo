# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
# Import functions/classes to make the public API
"""
Choclo: Kernel functions for your geophysical models.

Choclo is a Python library that hosts optimized forward modelling and kernel
functions for running geophysical forward and inverse models, intended to be
used by other libraries as the underlying layer of their computation.
"""

from . import dipole, point, prism
from ._version import __version__

# Append a leading "v" to the generated version by setuptools_scm
__version__ = f"v{__version__}"
