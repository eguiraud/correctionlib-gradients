# SPDX-FileCopyrightText: 2023-present Enrico Guiraud <enrico.guiraud@pm.me>
#
# SPDX-License-Identifier: BSD-3-Clause
from functools import partial
from typing import TypeAlias

import correctionlib.schemav2 as schema
import jax
import numpy as np

# TODO: switch to use numpy.array_api.Array as _the_ array type.
# Must wait for it to be out of experimental.
# See https://numpy.org/doc/stable/reference/array_api.html.
Value: TypeAlias = float | np.ndarray | jax.Array


def apply_ast(ast: schema.Content, correction_name: str, _inputs: dict[str, Value]) -> Value:
    match ast:
        case float(x):
            return x
        case _:
            msg = (
                f"Cannot compute gradients of correction '{correction_name}': "
                f"it contains the unsupported operation type '{type(ast).__name__}'"
            )
            raise NotImplementedError(msg)


class CorrectionWithGradient:
    def __init__(self, c: schema.Correction):
        self._evaluator = partial(apply_ast, c.data, c.name)
        self._input_names = [v.name for v in c.inputs]
        self._name = c.name

    def evaluate(self, *inputs: Value) -> Value:
        if (n_in := len(inputs)) != (n_expected := len(self._input_names)):
            msg = f"This correction requires {n_expected} input(s), {n_in} provided"
            raise ValueError(msg)

        input_dict = dict(zip(self._input_names, inputs))
        return self._evaluator(input_dict)

    def eval_dict(self, inputs: dict[str, Value]) -> Value:
        for n in self._input_names:
            if n not in inputs:
                msg = f"Variable '{n}' is required by correction '{self._name}' but is not present in input"
                raise ValueError(msg)

        return self._evaluator(inputs)
