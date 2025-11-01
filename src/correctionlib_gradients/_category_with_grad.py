from typing import Union, Dict, List, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import correctionlib.schemav2 as schema

from correctionlib_gradients._formuladag import FormulaDAG
from correctionlib_gradients._utils import get_result_size
from correctionlib_gradients._typedefs import Value


@dataclass
class CategoryWithGrad:
    """A JAX-friendly representation of a Category correction."""
    var: str  # The category input variable
    content: Dict[Any, Union[float, 'FormulaDAG', 'CategoryWithGrad']]  # Map from original keys to value/formula/category
    input_vars: List[str]  # All input variables needed
    default: Union[float, None]
    _key_to_idx: Dict[Any, int]  # Map from external keys (str/int) to internal indices
    _idx_to_key: Dict[int, Any]  # Map from internal indices to external keys

    @staticmethod
    def from_category(category: schema.Category, inputs: List[schema.Variable], generic_formulas: Dict[str, schema.Formula] = None) -> "CategoryWithGrad":
        content = {}
        key_to_idx = {}
        idx_to_key = {}
        
        # Sort the keys to ensure consistent ordering
        sorted_items = sorted(category.content, key=lambda x: x.key)
        
        for item in sorted_items:
            # Use the original key directly as the index
            idx = len(key_to_idx)
            key_to_idx[item.key] = idx
            idx_to_key[idx] = item.key
            
            if isinstance(item.value, float):
                content[item.key] = item.value
            elif isinstance(item.value, (schema.Formula, schema.FormulaRef)):
                if isinstance(item.value, schema.FormulaRef) and generic_formulas:
                    formula = generic_formulas[item.value.ref]
                else:
                    formula = item.value
                content[item.key] = FormulaDAG(formula, inputs)
            elif isinstance(item.value, schema.Category):
                # Recursively handle nested categories
                content[item.key] = CategoryWithGrad.from_category(item.value, inputs, generic_formulas)
            else:
                raise ValueError(f"Unsupported content type in Category: {type(item.value)}")
        
        return CategoryWithGrad(
            var=category.input,
            content=content,
            input_vars=[v.name for v in inputs],
            default=getattr(category, 'default', None),
            _key_to_idx=key_to_idx,
            _idx_to_key=idx_to_key
        )

    def evaluate(self, inputs: Dict[str, Value]) -> Value:
        """Evaluate the category correction for the given inputs."""
        orig_x = inputs[self.var]  # Get the category selector value
        
        # Convert JAX array to Python value if needed
        if isinstance(orig_x, (jax.Array, jnp.ndarray)):
            lookup_key = orig_x.item()
        else:
            lookup_key = orig_x
            
        if lookup_key not in self._key_to_idx:
            if self.default is not None:
                return jnp.array(self.default)
            raise ValueError(f"Category key '{lookup_key}' not found and no default specified")
        
        # Convert to internal index for JAX compatibility
        x = jnp.array(self._key_to_idx[lookup_key])
        
        def _handle_single_input(xi: Value, *args: Value) -> Value:
            # Convert back to original key for content lookup
            idx = int(xi)
            orig_key = self._idx_to_key[idx]
            value = self.content[orig_key]
            
            if isinstance(value, float):
                return value
            elif isinstance(value, CategoryWithGrad):
                # For nested categories, ensure we pass through only the inputs they need
                needed_inputs = {k: v for k, v in inputs.items() if k in value.input_vars}
                return value.evaluate(needed_inputs)
            else:  # FormulaDAG
                # Create input dict for formula evaluation, checking that we have all needed variables
                input_dict = {}
                # We need all variables that the formula depends on
                for name, arg in zip(self.input_vars, args):
                    input_dict[name] = arg
                # Also include any other variables from the inputs that the formula might need
                for name in inputs:
                    if name not in input_dict:
                        input_dict[name] = inputs[name]
                return value.evaluate(input_dict)

        # Get all required inputs as arrays except the category variable
        other_inputs = [inputs[name] for name in self.input_vars if name != self.var]
        
        # Handle both scalar and array inputs
        if jnp.isscalar(x) or (isinstance(x, jax.Array) and x.ndim == 0):
            return _handle_single_input(x, *other_inputs)
        else:
            # Vectorize the function using jax.vmap for array inputs
            vectorized_handler = jax.vmap(_handle_single_input)
            return vectorized_handler(x, *other_inputs)

    def __call__(self, inputs: Dict[str, Value]) -> Value:
        """Alias for evaluate()."""
        return self.evaluate(inputs)
