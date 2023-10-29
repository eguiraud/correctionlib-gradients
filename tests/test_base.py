# SPDX-FileCopyrightText: 2023-present Enrico Guiraud <enrico.guiraud@pm.me>
#
# SPDX-License-Identifier: BSD-3-Clause
import math
import re

import jax
import numpy as np
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
        output=schemav2.Variable(name="weight", type="real"),
        data=1.234,
    ),
    "simple-uniform-binning": schemav2.Correction(
        name="simple uniform binning",
        version=2,
        inputs=[schemav2.Variable(name="x", type="real")],
        output=schemav2.Variable(name="weight", type="real"),
        data=schemav2.Binning(
            nodetype="binning",
            input="x",
            edges=schemav2.UniformBinning(n=10, low=0.0, high=10.0),
            content=[1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            flow="clamp",
        ),
    ),
    "simple-nonuniform-binning": schemav2.Correction(
        name="simple non-uniform binning",
        version=2,
        inputs=[schemav2.Variable(name="x", type="real")],
        output=schemav2.Variable(name="weight", type="real"),
        data=schemav2.Binning(
            nodetype="binning",
            input="x",
            edges=[0.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0],
            content=[1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            flow="clamp",
        ),
    ),
    "compound-nonuniform-binning": schemav2.Correction(
        name="compound non-uniform binning",
        version=2,
        inputs=[schemav2.Variable(name="x", type="real")],
        output=schemav2.Variable(name="weight", type="real"),
        data=schemav2.Binning(
            nodetype="binning",
            input="x",
            edges=[0.0, 1.0],
            content=[schemav2.Formula(nodetype="formula", expression="x*x", parser="TFormula", variables=["x"])],
            flow="clamp",
        ),
    ),
    "simple-nonuniform-binning-flow-default": schemav2.Correction(
        name="simple non-uniform binning with a default value as 'flow'",
        version=2,
        inputs=[schemav2.Variable(name="x", type="real")],
        output=schemav2.Variable(name="weight", type="real"),
        data=schemav2.Binning(
            nodetype="binning",
            input="x",
            edges=[0.0, 1.0],
            content=[1.0],
            flow=42.0,
        ),
    ),
    # this type of correction is unsupported
    "categorical": schemav2.Correction(
        name="categorical",
        version=2,
        inputs=[schemav2.Variable(name="c", type="int")],
        output=schemav2.Variable(name="weight", type="real"),
        data=schemav2.Category(
            nodetype="category",
            input="x",
            content=[schemav2.CategoryItem(key=0, value=1.234)],
        ),
    ),
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


def test_unsupported_correction():
    with pytest.raises(ValueError, match="Correction 'categorical' contains the unsupported operation type 'Category'"):
        CorrectionWithGradient(schemas["categorical"])


def test_unsupported_compound_binning():
    with pytest.raises(
        ValueError,
        match=(
            re.escape(
                "Correction 'compound non-uniform binning' contains a compound "
                "Binning correction (one or more of the bin contents are not "
                "simple scalars). This is not supported."
            )
        ),
    ):
        CorrectionWithGradient(schemas["compound-nonuniform-binning"])


def test_unsupported_flow_type():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Correction 'simple non-uniform binning with a default value as "
            "'flow'' contains a Binning correction with `flow=42.0`. "
            "Only 'clamp' is supported."
        ),
    ):
        CorrectionWithGradient(schemas["simple-nonuniform-binning-flow-default"])


@pytest.mark.parametrize("jit", [False, True])
def test_evaluate_scale(jit):
    cg = CorrectionWithGradient(schemas["scale"])
    evaluate = jax.jit(cg.evaluate) if jit else cg.evaluate
    value, grad = jax.value_and_grad(evaluate)(4.2)
    assert math.isclose(value, 1.234)
    assert grad == 0.0


@pytest.mark.parametrize("jit", [False, True])
def test_eval_dict_scale(jit):
    cg = CorrectionWithGradient(schemas["scale"])
    eval_dict = jax.jit(cg.eval_dict) if jit else cg.eval_dict
    value, grad = jax.value_and_grad(eval_dict)({"x": 4.2})
    assert math.isclose(value, 1.234)
    assert list(grad.keys()) == ["x"]
    assert grad["x"] == 0.0


@pytest.mark.parametrize("jit", [False, True])
def test_vectorized_evaluate_scale(jit):
    cg = CorrectionWithGradient(schemas["scale"])
    evaluate = jax.jit(cg.evaluate) if jit else cg.evaluate
    x = np.array([0.0, 1.0])
    values, grads = jax.value_and_grad(evaluate)(x)
    assert np.allclose(values, [1.234, 1.234])
    assert len(grads) == 2
    assert grads[0] == 0.0
    assert grads[1] == 0.0


@pytest.mark.parametrize("jit", [False, True])
def test_vectorized_eval_dict_scale(jit):
    cg = CorrectionWithGradient(schemas["scale"])
    eval_dict = jax.jit(cg.eval_dict) if jit else cg.eval_dict
    x = np.array([0.0, 1.0])
    values, grads = jax.value_and_grad(eval_dict)({"x": x})
    assert np.allclose(values, [1.234, 1.234])
    assert list(grads.keys()) == ["x"]
    assert len(grads["x"]) == 2
    assert grads["x"][0] == 0.0
    assert grads["x"][1] == 0.0


def test_vectorized_evaluate_simple_uniform_binning():
    cg = CorrectionWithGradient(schemas["simple-uniform-binning"])
    x = [3.0, 5.0, 11.0]  # 11. overflows: it tests clamping

    values = cg.evaluate(x)
    # here and below, the magic numbers have been checked by plotting
    # the bins and their contents, the corresponding spline, and its derivative.
    assert np.allclose(values, [3.47303922, 5.15686275, 1.0])

    grads = np.vectorize(jax.grad(cg.evaluate))(x)
    expected_grad = [0.995098039, 0.0, 0.0]
    assert np.allclose(grads, expected_grad)


def test_vectorized_evaluate_simple_nonuniform_binning():
    cg = CorrectionWithGradient(schemas["simple-nonuniform-binning"])
    x = [3.0, 5.0, 11.0]  # 11. overflows: it tests clamping

    values = cg.evaluate(x)
    assert np.allclose(values, [2.0, 3.08611111, 1])

    grads = np.vectorize(jax.grad(cg.evaluate))(x)
    expected_grad = [0.794444444, 0.0, 0.0]
    assert np.allclose(grads, expected_grad)
