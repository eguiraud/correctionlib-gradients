# SPDX-FileCopyrightText: 2023-present Enrico Guiraud <enrico.guiraud@pm.me>
#
# SPDX-License-Identifier: BSD-3-Clause
from functools import partial
from typing import Any, Callable, TypeAlias

import jax
import numpy as np
from correctionlib.schemav2 import Content, Correction

Input: TypeAlias = float
Output: TypeAlias = tuple[float, dict[str, float]]


def isfloat(x: Any) -> bool:
    if isinstance(x, float):
        return True
    if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.floating):
        return True  # covers numpy and jax arrays
    return False


def apply_ast(ast: Content, correction_name: str, _inputs: dict[str, Input]):
    match ast:
        case float(x):
            return x
        case _:
            msg = (
                f"Cannot compute gradients of correction '{correction_name}': "
                f"it contains the unsupported operation type '{type(ast).__name__}'"
            )
            raise NotImplementedError(msg)


def grad_wrt_key(f: Callable[[dict[str, Input]], float], wrt: list[str] | None):
    """Given a callable f that takes a dictionary inputs, return a callable
    that evaluates the gradient of f w.r.t. one or more of the arrays in
    the dictionary (specified by key).

    If wrt is None, gradients are taken w.r.t. all floating point inputs.
    """

    def grad_wrt_impl(inputs: dict[str, jax.Array]):
        in_vars = list(inputs.keys())
        true_wrt = wrt if wrt is not None else [k for k, v in inputs.items() if isfloat(v)]
        argnums = [in_vars.index(wrt_var) for wrt_var in true_wrt]

        def f_with_positionals(*args):
            args_as_dict = dict(zip(in_vars, args))
            return f(args_as_dict)

        # TODO should we cache df instead of calling value_and_grad at every invocation?
        df = jax.value_and_grad(f_with_positionals, argnums)
        value, grads = df(*inputs.values())
        return value, dict(zip(true_wrt, grads))

    return grad_wrt_impl


class CorrectionWithGradient:
    def __init__(self, c: Correction, *, wrt=None, jit=False):
        if len(c.inputs) == 0:
            msg = "Cannot take gradients of a correction that has no input variables"
            raise ValueError(msg)

        self._grad_evaluator = grad_wrt_key(partial(apply_ast, c.data, c.name), wrt=wrt)
        if jit:
            self._grad_evaluator = jax.jit(self._grad_evaluator)
        self._input_names = [v.name for v in c.inputs]
        self._name = c.name

    def evaluate(self, inputs: dict[str, Input]) -> Output:
        for n in self._input_names:
            if n not in inputs:
                msg = f"Variable '{n}' is required by correction '{self._name}' but is not present in input"
                raise ValueError(msg)

        value, grads = self._grad_evaluator(inputs)
        return (float(value), grads)
