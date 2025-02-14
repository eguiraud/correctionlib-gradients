import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from correctionlib import schemav2

from correctionlib_gradients import CorrectionWithGradient


schemas = {
    "categorical-with-formula": schemav2.Correction(
        name="categorical-with-formula",
        version=2,
        inputs=[
            schemav2.Variable(name="x", type="real"),
            schemav2.Variable(name="c", type="int"),
        ],
        output=schemav2.Variable(name="a scale", type="real"),
        data=schemav2.Category(
            nodetype="category",
            input="c",
            content=[
                schemav2.CategoryItem(
                    key=1,
                    value=schemav2.Formula(
                        nodetype="formula",
                        variables=["x"],
                        parser="TFormula",
                        expression="42*x",
                    ),
                ),
                schemav2.CategoryItem(
                    key=-1,
                    value=schemav2.Formula(
                        nodetype="formula",
                        variables=["x"],
                        parser="TFormula",
                        expression="1337*x",
                    ),
                ),
            ]
        ),
    ),
    # can be differentiated w.r.t. x, but not c
    "str-categorical-with-formula": schemav2.Correction(
        name="str-categorical-with-formula",
        version=2,
        inputs=[
            schemav2.Variable(name="x", type="real"),
            schemav2.Variable(name="c", type="string"),
        ],
        output=schemav2.Variable(name="a scale", type="real"),
        data=schemav2.Category(
            nodetype="category",
            input="c",
            content=[
                schemav2.CategoryItem(
                    key="up",
                    value=schemav2.Formula(
                        nodetype="formula",
                        variables=["x"],
                        parser="TFormula",
                        expression="42*x",
                    ),
                ),
                schemav2.CategoryItem(
                    key="down",
                    value=schemav2.Formula(
                        nodetype="formula",
                        variables=["x"],
                        parser="TFormula",
                        expression="1337*x",
                    ),
                ),
            ]
        ),
    ),
    # this type of correction is unsupported
    "compound-binning-with-categorical": schemav2.Correction(
        name="compound binning with categorical",
        version=2,
        inputs=[schemav2.Variable(name="x", type="real")],
        output=schemav2.Variable(name="weight", type="real"),
        data=schemav2.Binning(
            nodetype="binning",
            input="x",
            edges=schemav2.UniformBinning(n=1, low=0.0, high=1.0),
            content=[
                schemav2.Category(nodetype="category", input="x", content=[schemav2.CategoryItem(key=0, value=1.234)])
            ],
            flow="clamp",
        ),
    ),
}

cg = CorrectionWithGradient(schemas["categorical-with-formula"])
value = cg.evaluate(0.5, 1)
assert math.isclose(value, 21.0)

value = cg.evaluate(0.5, -1)
assert math.isclose(value, 668.5)


cg = CorrectionWithGradient(schemas["str-categorical-with-formula"])
value = cg.evaluate(0.5, "up")
assert math.isclose(value, 21.0)

value = cg.evaluate(0.5, "down")
assert math.isclose(value, 668.5)


# cg = CorrectionWithGradient(schemas["str-categorical-with-formula"])
# value = cg.evaluate(0.5, "up")
# assert math.isclose(value, 21.0)
