# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Kernels and forward modelling functions for rectangular prisms
"""
from ._forward import gravity_e, gravity_ee, gravity_n, gravity_pot, gravity_u
from ._kernels import kernel_e, kernel_ee, kernel_n, kernel_pot, kernel_u
