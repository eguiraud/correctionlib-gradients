# SPDX-FileCopyrightText: 2023-present Enrico Guiraud <enrico.guiraud@pm.me>
#
# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import dataclass
from typing import Callable

import jax
import numpy as np
from correctionlib.schemav2 import Binning, UniformBinning
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]

from correctionlib_gradients._typedefs import Value


def _midpoints(x: jax.Array) -> jax.Array:
    return 0.5 * (x[1:] + x[:-1])


@dataclass
class SplineWithGrad:
    spline: Callable[[Value], Value]
    var: str

    def __call__(self, v: Value) -> Value:
        return self.spline(v)


# TODO: the callable returned is not traceable by JAX, so it does not support jax.jit, jax.vmap etc.
# TODO: would it make more sense if the value returned was the exact value given by the binning,
#       while the derivative is calculated by spline.derivative?
def make_differentiable_spline(b: Binning) -> SplineWithGrad:
    var: str = b.input

    if isinstance(b.edges, UniformBinning):
        xs = np.linspace(b.edges.low, b.edges.high, b.edges.n + 1)
    else:
        xs = np.array(b.edges)

    ys = b.content
    assert all(isinstance(y, float) for y in ys)  # noqa: S101

    spline = CubicSpline(_midpoints(xs), ys, bc_type="clamped")
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

    return SplineWithGrad(spline=eval_spline, var=var)
