# SPDX-FileCopyrightText: 2023-present Enrico Guiraud <enrico.guiraud@pm.me>
#
# SPDX-License-Identifier: BSD-3-Clause
import correctionlib.schemav2 as schema
import jax
import jax.numpy as jnp
import numpy as np

from correctionlib_gradients._correctiondag import CorrectionDAG
from correctionlib_gradients._typedefs import Value


class CorrectionWithGradient:
    def __init__(self, c: schema.Correction):
        self._dag = CorrectionDAG(c)
        self._input_vars = c.inputs
        self._name = c.name

    def evaluate(self, *inputs: Value) -> jax.Array:
        self._check_num_inputs(inputs)
        inputs_as_jax = tuple(jnp.array(i) for i in inputs)
        self._check_input_types(inputs_as_jax)
        input_names = (v.name for v in self._input_vars)

        input_dict = dict(zip(input_names, inputs_as_jax))
        return self._dag.evaluate(input_dict)

    def _check_num_inputs(self, inputs: tuple[Value, ...]) -> None:
        if (n_in := len(inputs)) != (n_expected := len(self._input_vars)):
            msg = (
                f"This correction requires {n_expected} input(s), {n_in} provided."
                f" Required inputs are {[v.name for v in self._input_vars]}"
            )
            raise ValueError(msg)

    def _check_input_types(self, inputs: tuple[jax.Array, ...]) -> None:
        for i, v in enumerate(inputs):
            in_type = v.dtype
            expected_type_str = self._input_vars[i].type
            expected_type = {"real": np.floating, "int": np.integer}[expected_type_str]
            if not np.issubdtype(in_type, expected_type):
                msg = (
                    f"Variable '{self._input_vars[i].name}' has type {in_type}"
                    f" instead of the expected {expected_type.__name__}"
                )
                raise ValueError(msg)
