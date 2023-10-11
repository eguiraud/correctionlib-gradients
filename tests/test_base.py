# SPDX-FileCopyrightText: 2023-present Enrico Guiraud <enrico.guiraud@pm.me>
#
# SPDX-License-Identifier: BSD-3-Clause
import math

import jax
import pytest
from correctionlib import schemav2

from correctionlib_gradients import CorrectionWithGradient

# we need to set this to avoid loss of precision when jit=True
# due to JAX aggressively casting everything to float32.
jax.config.update("jax_enable_x64", True)

schemas = {
    "scale": schemav2.Correction(
        name="test scalar",
        version=2,
        inputs=[schemav2.Variable(name="x", type="real")],
        output=schemav2.Variable(name="a scale", type="real"),
        data=1.234,
    )
}


def test_wrong_input_length():
    cg = CorrectionWithGradient(schemas["scale"])

    with pytest.raises(ValueError, match="This correction requires 1 input\\(s\\), 2 provided"):
        cg.evaluate(0.0, 1.0)


def test_missing_input():
    cg = CorrectionWithGradient(schemas["scale"])

    with pytest.raises(
        ValueError, match="Variable 'x' is required by correction 'test scalar' but is not present in input"
    ):
        cg.eval_dict({})


@pytest.mark.parametrize("jit", [False, True])
def test_scale(jit) -> None:
    cg = CorrectionWithGradient(schemas["scale"])

    # evaluate
    evaluate = jax.jit(cg.evaluate) if jit else cg.evaluate
    value, grad = jax.value_and_grad(evaluate)(4.2)
    assert math.isclose(value, 1.234)
    assert grad.item() == 0.0

    # eval_dict
    eval_dict = jax.jit(cg.eval_dict) if jit else cg.eval_dict
    value, grad = jax.value_and_grad(eval_dict)({"x": 4.2})
    assert math.isclose(value, 1.234)
    assert list(grad.keys()) == ["x"]
    assert grad["x"] == 0.0
