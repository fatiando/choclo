# Copyright (c) 2022 The Choclo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Kernels and forward modelling functions for rectangular prisms
"""
from ._gravity import (
    gravity_e,
    gravity_ee,
    gravity_en,
    gravity_eu,
    gravity_n,
    gravity_nn,
    gravity_nu,
    gravity_pot,
    gravity_u,
    gravity_uu,
)
from ._kernels import (
    kernel_e,
    kernel_ee,
    kernel_eee,
    kernel_een,
    kernel_eeu,
    kernel_en,
    kernel_enn,
    kernel_enu,
    kernel_eu,
    kernel_euu,
    kernel_n,
    kernel_nn,
    kernel_nnn,
    kernel_nnu,
    kernel_nu,
    kernel_nuu,
    kernel_pot,
    kernel_u,
    kernel_uu,
    kernel_uuu,
)
from ._magnetic import (
    magnetic_e,
    magnetic_ee,
    magnetic_en,
    magnetic_eu,
    magnetic_field,
    magnetic_n,
    magnetic_nn,
    magnetic_nu,
    magnetic_u,
    magnetic_uu,
)
