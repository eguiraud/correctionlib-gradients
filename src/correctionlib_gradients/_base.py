# SPDX-FileCopyrightText: 2023-present Enrico Guiraud <enrico.guiraud@pm.me>
#
# SPDX-License-Identifier: BSD-3-Clause
from typing import Callable, Iterable, TypeAlias, cast

import correctionlib.schemav2 as schema
import jax
import numpy as np
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]

from correctionlib_gradients._typedefs import Array, Value


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


DAGNode: TypeAlias = float | schema.Binning


class CorrectionDAG:
    """A JAX-friendly representation of a correctionlib.schemav2.Correction's DAG."""

    input_vars: list[schema.Variable]
    node: DAGNode

    def __init__(self, c: schema.Correction):
        """Transform a schemav2.Correction object in a JAX-friendly DAG.

        Errors out in case the correction contains unsupported operations.

        Transformations applied:
        - correctionlib.schema.Formula -> FormulaAST, a JAX-friendly formula evaluator object.
        - [TODO] Binning nodes with constant bin contents -> differentiable relaxation.
        """
        self.input_vars = c.inputs
        match c.data:
            case float(x):
                self.node = x
            case schema.Binning(content=[*values], flow="clamp"):
                if not all(isinstance(v, float) for v in values):  # type: ignore[has-type]
                    msg = (
                        f"Correction '{c.name}' contains a compound Binning correction"
                        "(one or more of the bin contents are not simple scalars). This is not supported."
                    )
                    raise ValueError(msg)
                self.node = c.data
            case schema.Binning(flow=flow):
                flow = cast(str, flow)  # type: ignore[has-type]
                msg = f"Correction '{c.name}' contains a Binning correction with `{flow=}`. Only 'clamp' is supported."
                raise ValueError(msg)
            case _:
                msg = f"Correction '{c.name}' contains the unsupported operation type '{type(c.data).__name__}'"
                raise ValueError(msg)

    def evaluate(self, inputs: dict[str, Value]) -> Value:
        match self.node:
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


class CorrectionWithGradient:
    def __init__(self, c: schema.Correction):
        self._dag = CorrectionDAG(c)
        self._input_names = [v.name for v in c.inputs]
        self._name = c.name

    def evaluate(self, *inputs: Value) -> Value:
        if (n_in := len(inputs)) != (n_expected := len(self._input_names)):
            msg = f"This correction requires {n_expected} input(s), {n_in} provided"
            raise ValueError(msg)

        input_dict = dict(zip(self._input_names, inputs))
        return self._dag.evaluate(input_dict)

    def eval_dict(self, inputs: dict[str, Value]) -> Value:
        for n in self._input_names:
            if n not in inputs:
                msg = f"Variable '{n}' is required by correction '{self._name}' but is not present in input"
                raise ValueError(msg)

        return self._dag.evaluate(inputs)
