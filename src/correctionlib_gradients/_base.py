# SPDX-FileCopyrightText: 2023-present Enrico Guiraud <enrico.guiraud@pm.me>
#
# SPDX-License-Identifier: BSD-3-Clause
from functools import partial
from typing import Callable, Iterable, TypeAlias, cast

import correctionlib.schemav2 as schema
import jax
import numpy as np
from scipy.interpolate import CubicSpline  # type: ignore[import-not-found]

# TODO: switch to use numpy.array_api.Array as _the_ array type.
# Must wait for it to be out of experimental.
# See https://numpy.org/doc/stable/reference/array_api.html.
Array: TypeAlias = np.ndarray | jax.Array
Value: TypeAlias = float | Array


def midpoints(x: Array) -> Array:
    return 0.5 * (x[1:] + x[:-1])


# TODO: the callable returned is not traceable by JAX, so it does not support jax.jit, jax.vmap etc.
# TODO: would it make more sense if the value returned was the exact value given by the binning,
#       while the derivative is calculated by spline.derivative?
def make_differentiable_spline(x: Array, y: Array) -> Callable[[Value], Value]:
    spline = CubicSpline(midpoints(x), y, bc_type="clamped")
    dspline = spline.derivative(1)

    def clip(x: Value) -> Value:
        # so that extrapolation works
        return np.clip(x, spline.x[0], spline.x[-1])

    @jax.custom_vjp
    def eval_spline(x):  # type: ignore[no-untyped-def]
        return spline(clip(x))

    def eval_spline_fwd(x):  # type: ignore[no-untyped-def]
        return eval_spline(x), dspline(clip(x))

    def eval_spline_bwd(res, g):  # type: ignore[no-untyped-def]
        return ((res * g),)

    eval_spline.defvjp(eval_spline_fwd, eval_spline_bwd)

    return cast(Callable[[Value], Value], eval_spline)


def apply_ast(ast: schema.Content, inputs: dict[str, Value]) -> Value:
    match ast:
        case float(x):
            return x
        case schema.Binning(edges=_edges, content=[*_values], input=_var, flow="clamp"):
            # to make mypy happy
            var: str = _var  # type: ignore[has-type]
            edges: Iterable[float] | schema.UniformBinning = _edges  # type: ignore[has-type]
            values: Array = _values  # type: ignore[has-type]

            if isinstance(edges, schema.UniformBinning):
                xs = np.linspace(edges.low, edges.high, edges.n + 1)
            else:
                xs = np.array(edges)
            s = make_differentiable_spline(xs, values)
            return s(inputs[var])
        case _:  # pragma: no cover
            msg = "Unsupported type of node in the computation graph. This should never happen."
            raise RuntimeError(msg)


def assert_supported(c: schema.Correction, name: str) -> None:
    match c.data:
        case float():
            pass
        case schema.Binning(content=[*values], flow="clamp"):
            if not all(isinstance(v, float) for v in values):  # type: ignore[has-type]
                msg = (
                    f"Correction '{name}' contains a compound Binning correction"
                    "(one or more of the bin contents are not simple scalars). This is not supported."
                )
                raise ValueError(msg)
        case schema.Binning(flow=flow):
            flow = cast(str, flow)  # type: ignore[has-type]
            msg = f"Correction '{name}' contains a Binning correction with `{flow=}`. Only 'clamp' is supported."
            raise ValueError(msg)
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
