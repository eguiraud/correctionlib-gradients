# SPDX-FileCopyrightText: 2023-present Enrico Guiraud <enrico.guiraud@pm.me>
#
# SPDX-License-Identifier: BSD-3-Clause
import jax


def get_result_size(inputs: dict[str, jax.Array]) -> int:
    """Calculate what size the result of a DAG evaluation should have.

    The size is equal to the one, common size (shape[0], or number or rows) of all
    the non-scalar inputs we require, or 0 if all inputs are scalar.
    An error is thrown in case the shapes of two non-scalar inputs differ.
    """
    result_shape: tuple[int, ...] = ()
    for value in inputs.values():
        if result_shape == ():
            result_shape = value.shape
        elif value.shape != result_shape:
            msg = "The shapes of all non-scalar inputs should match."
            raise ValueError(msg)
    if result_shape != ():
        return result_shape[0]
    else:
        return 0
