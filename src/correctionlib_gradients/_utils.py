# SPDX-FileCopyrightText: 2023-present Enrico Guiraud <enrico.guiraud@pm.me>
#
# SPDX-License-Identifier: BSD-3-Clause
from typing import Union
import jax


def get_result_size(inputs: dict[str, Union[jax.Array, str]]) -> int:
    """Calculate what size the result of a DAG evaluation should have.
    The size is equal to the one, common size (shape[0], or number or rows) of all
    the non-scalar numeric inputs we require, or 0 if all inputs are scalar or strings.
    An error is thrown in case the shapes of two non-scalar inputs differ.
    """
    result_shape: tuple[int, ...] = ()
    for value in inputs.values():
        # Skip string inputs when determining result size
        if isinstance(value, str):
            continue
        if result_shape == ():
            result_shape = value.shape
        elif value.shape != result_shape:
            msg = "The shapes of all non-scalar inputs should match."
            raise ValueError(msg)
    if result_shape != ():
        return result_shape[0]
    else:
        return 0