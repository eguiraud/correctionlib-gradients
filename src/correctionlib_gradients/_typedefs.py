from typing import TypeAlias

import jax
import numpy as np

# TODO: switch to use numpy.array_api.Array as _the_ array type.
# Must wait for it to be out of experimental.
# See https://numpy.org/doc/stable/reference/array_api.html.
Array: TypeAlias = np.ndarray | jax.Array
Value: TypeAlias = float | Array
