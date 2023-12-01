# correctionlib-gradients

[![ci](https://github.com/eguiraud/correctionlib-gradients/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/eguiraud/correctionlib-gradients/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/eguiraud/correctionlib-gradients/graph/badge.svg?token=T4F8CB1DYN)](https://codecov.io/gh/eguiraud/correctionlib-gradients)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/eguiraud/correctionlib-gradients/main.svg)](https://results.pre-commit.ci/latest/github/eguiraud/correctionlib-gradients/main)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI - Version](https://img.shields.io/pypi/v/correctionlib-gradients.svg)](https://pypi.org/project/correctionlib-gradients)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/correctionlib-gradients.svg)](https://pypi.org/project/correctionlib-gradients)

A [JAX](https://jax.readthedocs.io)-friendly, auto-differentiable, Python-only implementation of [correctionlib](https://github.com/cms-nanoAOD/correctionlib) correction evaluations.

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

1. construct a `CorrectionWithGradient` object from a `correctionlib.schemav2.Correction`
2. there is no point 2: you can use `CorrectionWithGradient.evaluate` as a normal JAX-friendly, auto-differentiable function

### Example

```python
import jax
import jax.numpy as jnp

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
assert jnp.isclose(value, 9.0)
assert jnp.isclose(grad, 6.0)

# for Formula corrections, jax.jit and jax.vmap work too
xs = jnp.array([3.0, 4.0])
values, grads = jax.vmap(jax.jit(jax.value_and_grad(c.evaluate)))(xs)
assert jnp.allclose(values, jnp.array([9.0, 16.0]))
assert jnp.allclose(grads, jnp.array([6.0, 8.0]))
```

## Supported types of corrections

Currently the following corrections from `correctionlib.schemav2` are supported:

- `Formula`, including parametrical formulas
- `Binning` with uniform or non-uniform bin edges and `flow="clamp"`; bin contents can be either:
  - all scalar values
  - all `Formula` or `FormulaRef`
- scalar constants

## License

`correctionlib-gradients` is distributed under the terms of the [BSD 3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) license.
