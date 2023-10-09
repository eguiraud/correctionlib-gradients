# SPDX-FileCopyrightText: 2023-present Enrico Guiraud <enrico.guiraud@pm.me>
#
# SPDX-License-Identifier: BSD-3-Clause
import math

from correctionlib import schemav2
from jax import config

from correctionlib_gradients import CorrectionWithGradient

# we need to set this to avoid loss of precision when jit=True
# due to JAX aggressively casting everything to float32.
config.update("jax_enable_x64", True)


def test_scalar():
    c = schemav2.Correction(
            name="test scalar",
            version=2,
            inputs=[schemav2.Variable(name="x", type="real")],
            output=schemav2.Variable(name="a scale", type="real"),
            data=1.234,
        )

    dc = CorrectionWithGradient(c)
    value, gradients = dc.evaluate({"x": 4.2})
    assert math.isclose(value, 1.234)
    assert list(gradients.keys()) == ["x"]
    assert gradients["x"] == 0.

    # with an extra "y" input of string type
    dc = CorrectionWithGradient(c, wrt=["x"])
    value, gradients = dc.evaluate({"x": 4.2, "y": "another"})
    assert math.isclose(value, 1.234)
    assert list(gradients.keys()) == ["x"]
    assert gradients["x"] == 0.

    # with jit=True
    # the extra "y" variable cannot be a string, jax.jit cannot handle it
    dc = CorrectionWithGradient(c, wrt=["x"], jit=True)
    value, gradients = dc.evaluate({"x": 4.2, "y": 42})
    assert math.isclose(value, 1.234)
    assert list(gradients.keys()) == ["x"]
    assert gradients["x"] == 0.

