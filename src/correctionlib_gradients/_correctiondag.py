from typing import TypeAlias, cast

import correctionlib.schemav2 as schema
import jax
import jax.numpy as jnp

import correctionlib_gradients._utils as utils
from correctionlib_gradients._compound_binning import CompoundBinning
from correctionlib_gradients._formuladag import FormulaDAG
from correctionlib_gradients._spline_with_grad import SplineWithGrad

DAGNode: TypeAlias = float | SplineWithGrad | FormulaDAG | CompoundBinning


class CorrectionDAG:
    """A JAX-friendly representation of a correctionlib.schemav2.Correction's DAG."""

    input_names: list[str]
    node: DAGNode

    def __init__(self, c: schema.Correction):
        """Transform a schemav2.Correction object in a JAX-friendly DAG.

        Errors out in case the correction contains unsupported operations.

        Transformations applied:
        - correctionlib.schema.Formula -> FormulaAST, a JAX-friendly formula evaluator object.
        - Binning nodes with constant bin contents -> differentiable relaxation.
        """
        self.input_names = [v.name for v in c.inputs]
        match c.data:
            case float(x):
                self.node = x
            case schema.Binning(content=[*values], flow="clamp") as binning:
                if all(isinstance(v, float) for v in values):  # type: ignore[has-type]
                    # simple binning
                    self.node = SplineWithGrad.from_binning(c.data)
                elif all(isinstance(v, (schema.Formula, schema.FormulaRef)) for v in values):  # type: ignore[has-type]
                    self.node = CompoundBinning(binning, c.generic_formulas)
                else:
                    msg = (
                        f"Correction '{c.name}' contains a Binning correction but the bin contents"
                        " are neither all scalars nor all Formulas/FormulaRefs. This is not supported."
                    )
                    raise ValueError(msg)
            case schema.Binning(flow=flow):
                flow = cast(str, flow)  # type: ignore[has-type]
                msg = f"Correction '{c.name}' contains a Binning correction with `{flow=}`. Only 'clamp' is supported."
                raise ValueError(msg)
            case schema.Formula() as f:
                self.node = FormulaDAG(f, c.inputs)
            case _:
                msg = f"Correction '{c.name}' contains the unsupported operation type '{type(c.data).__name__}'"
                raise ValueError(msg)

    def evaluate(self, inputs: dict[str, jax.Array]) -> jax.Array:
        result_size = utils.get_result_size(inputs)

        match self.node:
            case float(x):
                if result_size == 0:
                    return jnp.array(x)
                else:
                    return jnp.repeat(x, result_size)
            case SplineWithGrad() as s:
                return s(inputs[s.var])
            case FormulaDAG() as f:
                return f.evaluate(inputs)
            case CompoundBinning() as cb:
                return cb.evaluate(inputs)
