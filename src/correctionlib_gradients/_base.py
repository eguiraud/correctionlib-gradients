# SPDX-FileCopyrightText: 2023-present Enrico Guiraud <enrico.guiraud@pm.me>
#
# SPDX-License-Identifier: BSD-3-Clause
from typing import Callable, Iterable, TypeAlias, cast

import correctionlib.schemav2 as schema
import jax
import numpy as np
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]

from correctionlib_gradients._typedefs import Value


def midpoints(x: jax.Array) -> jax.Array:
    return 0.5 * (x[1:] + x[:-1])


# TODO: the callable returned is not traceable by JAX, so it does not support jax.jit, jax.vmap etc.
# TODO: would it make more sense if the value returned was the exact value given by the binning,
#       while the derivative is calculated by spline.derivative?
def make_differentiable_spline(x: jax.Array, y: jax.Array) -> Callable[[Value], Value]:
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

    input_names: list[str]
    node: DAGNode

    def __init__(self, c: schema.Correction):
        """Transform a schemav2.Correction object in a JAX-friendly DAG.

        Errors out in case the correction contains unsupported operations.

        Transformations applied:
        - correctionlib.schema.Formula -> FormulaAST, a JAX-friendly formula evaluator object.
        - [TODO] Binning nodes with constant bin contents -> differentiable relaxation.
        """
        self.input_names = [v.name for v in c.inputs]
        match c.data:
            case float(x):
                self.node = x
            case schema.Binning(content=[*values], flow="clamp"):
                if not all(isinstance(v, float) for v in values):  # type: ignore[has-type]
                    msg = (
                        f"Correction '{c.name}' contains a compound Binning correction"
                        " (one or more of the bin contents are not simple scalars). This is not supported."
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

    def evaluate(self, inputs: dict[str, jax.Array]) -> jax.Array:
        result_size = self._get_result_size(inputs)

        match self.node:
            case float(x):
                if result_size == 0:
                    return jax.numpy.array(x)
                else:
                    return jax.numpy.array([x] * result_size)
            case schema.Binning(edges=_edges, content=[*_values], input=_var, flow="clamp"):
                # to make mypy happy
                var: str = _var  # type: ignore[has-type]
                edges: Iterable[float] | schema.UniformBinning = _edges  # type: ignore[has-type]
                values: jax.Array = _values  # type: ignore[has-type]

                if isinstance(edges, schema.UniformBinning):
                    xs = np.linspace(edges.low, edges.high, edges.n + 1)
                else:
                    xs = np.array(edges)
                s = make_differentiable_spline(xs, values)
                return s(inputs[var])
            case _:  # pragma: no cover
                msg = "Unsupported type of node in the computation graph. This should never happen."
                raise RuntimeError(msg)

    def _get_result_size(self, inputs: dict[str, jax.Array]) -> int:
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


class CorrectionWithGradient:
    def __init__(self, c: schema.Correction):
        self._dag = CorrectionDAG(c)
        self._input_vars = c.inputs
        self._name = c.name

    def evaluate(self, *inputs: Value) -> jax.Array:
        self._check_num_inputs(inputs)
        inputs_as_jax = tuple(jax.numpy.array(i) for i in inputs)
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
