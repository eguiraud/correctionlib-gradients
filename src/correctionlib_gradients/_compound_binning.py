# SPDX-FileCopyrightText: 2023-present Enrico Guiraud <enrico.guiraud@pm.me>
#
# SPDX-License-Identifier: BSD-3-Clause
import correctionlib.schemav2 as schema
import jax
import jax.numpy as jnp

from correctionlib_gradients._formuladag import FormulaDAG


class CompoundBinning:
    var: str
    edges: jax.Array
    values: list[FormulaDAG]

    def __init__(self, b: schema.Binning):
        # nothing else is supported
        assert b.flow == "clamp"  # noqa: S101

        self.var = b.input

        if isinstance(b.edges, schema.UniformBinning):
            self.edges = jnp.linspace(b.edges.low, b.edges.high, b.edges.n + 1)
        else:
            self.edges = jnp.array(b.edges)

        self.values = []
        for value in b.content:
            assert isinstance(value, schema.Formula)  # noqa: S101
            variable = schema.Variable(name=self.var, type="real")
            formula = FormulaDAG(value, inputs=[variable])
            self.values.append(formula)

    def evaluate(self, inputs: dict[str, jax.Array]) -> jax.Array:
        x = inputs[self.var]
        assert x.shape == ()  # noqa: S101
        jnp.clip(x, self.edges[0], self.edges[-1])
        bin_idx = jnp.searchsorted(self.edges, x, side="right") - 1
        return self.values[bin_idx].evaluate(inputs)
