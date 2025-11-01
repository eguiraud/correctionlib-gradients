# SPDX-FileCopyrightText: 2023-present Enrico Guiraud <enrico.guiraud@pm.me>
#
# SPDX-License-Identifier: BSD-3-Clause
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import correctionlib.schemav2 as schema

from correctionlib_gradients._correctiondag import CorrectionDAG
from correctionlib_gradients._typedefs import Value


class CorrectionWithGradient:
    def __init__(self, c: schema.Correction):
        self._dag = CorrectionDAG(c)
        self._input_vars = c.inputs
        self._name = c.name

    def evaluate(self, *inputs: Value) -> jax.Array:
        self._check_num_inputs(inputs)
        inputs_as_jax = tuple(
            i if isinstance(i, str) else jnp.array(i)
            for i in inputs
        )
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

    def _check_input_types(self, inputs: tuple[Union[jax.Array, str], ...]) -> None:
        for i, v in enumerate(inputs):
            expected_type_str = self._input_vars[i].type
            
            if expected_type_str == "string":
                if not isinstance(v, str):
                    msg = (
                        f"Variable '{self._input_vars[i].name}' should be a string "
                        f"but got {type(v).__name__}"
                    )
                    raise ValueError(msg)
            else:
                # For numeric types, check the dtype
                if isinstance(v, str):
                    msg = (
                        f"Variable '{self._input_vars[i].name}' should be numeric "
                        f"but got a string"
                    )
                    raise ValueError(msg)
                    
                in_type = v.dtype
                expected_type = {"real": np.floating, "int": np.integer}[expected_type_str]
                if not np.issubdtype(in_type, expected_type):
                    msg = (
                        f"Variable '{self._input_vars[i].name}' has type {in_type}"
                        f" instead of the expected {expected_type.__name__}"
                    )
                    raise ValueError(msg)
