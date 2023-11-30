# SPDX-FileCopyrightText: 2023-present Enrico Guiraud <enrico.guiraud@pm.me>
#
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any, TypeAlias

import jax
import numpy.typing

# TODO: switch to use numpy.array_api.Array as _the_ array type.
# Must wait for it to be out of experimental.
# See https://numpy.org/doc/stable/reference/array_api.html.
Value: TypeAlias = float | numpy.typing.NDArray[Any] | jax.Array | list[float]
