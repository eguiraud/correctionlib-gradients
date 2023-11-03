# correctionlib-gradients

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/eguiraud/correctionlib-gradients/main.svg)](https://results.pre-commit.ci/latest/github/eguiraud/correctionlib-gradients/main)
[![ci](https://github.com/eguiraud/correctionlib-gradients/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/eguiraud/correctionlib-gradients/actions/workflows/test.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/eguiraud/correctionlib-gradients/graph/badge.svg?token=T4F8CB1DYN)](https://codecov.io/gh/eguiraud/correctionlib-gradients)
<br>
[![PyPI - Version](https://img.shields.io/pypi/v/correctionlib-gradients.svg)](https://pypi.org/project/correctionlib-gradients)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/correctionlib-gradients.svg)](https://pypi.org/project/correctionlib-gradients)

---

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Supported types of corrections](#supported-types-of-corrections)
- [License](#license)

## Installation

```console
pip install correctionlib-gradients
```

## Usage

1. construct a `CorrectionWithGradient` object from a correctionlib schema
2. there is no point 2: you can use `CorrectionWithGradient.evaluate` as a normal JAX-friendly, auto-differentiable function

```python
import jax

from correctionlib import schemav2
from correctionlib_gradients import CorrectionWithGradient

# given a correctionlib schema:
formula_schema = schemav2.Correction(
    name="x squared",
    version=2,
    inputs=[schemav2.Variable(name="x", type="real")],
    output=schemav2.Variable(name="a scale", type="real"),
    data=schemav2.Formula(
        nodetype="formula",
        expression="x * x",
        parser="TFormula",
        variables=["x"],
    ),
)

# construct a CorrectionWithGradient
c = CorrectionWithGradient(formula_schema)

# use c.evaluate as a JAX-friendly, auto-differentiable function
value, grad = jax.value_and_grad(c.evaluate)(3.0)
assert jax.numpy.isclose(value, 9.0)
assert jax.numpy.isclose(grad, 6.0)

# jax.jit works too
value, grad = jax.jit(jax.value_and_grad(c.evaluate))(3.0)
assert jax.numpy.isclose(value, 9.0)
assert jax.numpy.isclose(grad, 6.0)
```

## Supported types of corrections

Currently the following corrections from `correctionlib.schemav2` are supported:

- `Formula`
- `Binning` with uniform and non-uniform bin edges, simple scalar bin values, and `flow="clamp"`
- scalar constants

## License

`correctionlib-gradients` is distributed under the terms of the [BSD 3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) license.
