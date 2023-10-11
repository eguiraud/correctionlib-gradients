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


def apply_ast(ast: schema.Content, _inputs: dict[str, Value]) -> Value:
    match ast:
        case float(x):
            return x
        case _:  # pragma: no cover
            msg = "Unsupported type of node in the computation graph. This should never happen."
            raise RuntimeError(msg)


def assert_supported(c: schema.Correction, name: str) -> None:
    match c.data:
        case float():
            return
        case _:
            msg = f"Correction '{name}' contains the unsupported operation type '{type(c.data).__name__}'"
            raise ValueError(msg)


class CorrectionWithGradient:
    def __init__(self, c: schema.Correction):
        assert_supported(c, c.name)

        self._evaluator = partial(apply_ast, c.data)
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
