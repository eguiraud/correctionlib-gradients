from typing import Union, Dict, List
import jax
import jax.numpy as jnp
import correctionlib.schemav2 as schema
from correctionlib_gradients._formuladag import FormulaDAG
from correctionlib_gradients._utils import get_result_size
from correctionlib_gradients._typedefs import Value
from dataclasses import dataclass

from correctionlib_gradients._typedefs import Value


@dataclass
class CategoryWithGrad:
    var: str  # The category input variable (e.g. 'q')
    content: Dict[int, Union[float, 'FormulaDAG']]  # Map from category to value/formula
    input_vars: List[str]  # All input variables needed (e.g. ['phi', 'q'])
    default: Union[float, None]

    @staticmethod
    def from_category(category: schema.Category, inputs: List[schema.Variable], generic_formulas: Dict[str, schema.Formula] = None) -> "CategoryWithGrad":
        content = {}
        all_vars = {v.name for v in inputs}
        
        # Process each key-value pair in the category
        for item in category.content:
            if isinstance(item.value, float):
                content[item.key] = item.value
            elif isinstance(item.value, (schema.Formula, schema.FormulaRef)):
                # If it's a formula, convert it to FormulaDAG
                if isinstance(item.value, schema.FormulaRef) and generic_formulas:
                    formula = generic_formulas[item.value.ref]
                else:
                    formula = item.value
                content[item.key] = FormulaDAG(formula, inputs)
            else:
                raise ValueError(f"Unsupported content type in Category: {type(item.value)}")
        
        return CategoryWithGrad(
            var=category.input,
            content=content,
            input_vars=[v.name for v in inputs],
            default=getattr(category, 'default', None)
        )

    def evaluate(self, inputs: Dict[str, Value]) -> Value:
        """Evaluate the category correction for the given inputs."""
        x = inputs[self.var]  # Get the category selector value
        
        def _handle_single_input(xi: Value, *args: Value) -> Value:
            # Convert to int for category lookup
            key = int(xi)
            
            if key in self.content:
                value = self.content[key]
                if isinstance(value, float):
                    return value
                else:  # FormulaDAG
                    # Create input dict for formula evaluation
                    input_dict = {name: arg for name, arg in zip(self.input_vars, args)}
                    return value.evaluate(input_dict)
            elif self.default is not None:
                return self.default
            else:
                raise ValueError(f"Category key '{key}' not found and no default value specified")

        # Get all required inputs as arrays
        input_arrays = [inputs[name] for name in self.input_vars]
        
        # Handle both scalar and array inputs
        if jnp.isscalar(x) or (isinstance(x, jax.Array) and x.ndim == 0):
            return _handle_single_input(x, *input_arrays)
        else:
            # Vectorize the function using jax.vmap for array inputs
            vectorized_handler = jax.vmap(_handle_single_input)
            return vectorized_handler(x, *input_arrays)

    def __call__(self, inputs: Dict[str, Value]) -> Value:
        """Alias for evaluate()."""
        return self.evaluate(inputs)
