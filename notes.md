# Notes about correctionlib-gradients

Notes about specific quirks, warts, design decisions that might need revisiting, etc.
See [ARCHITECTURE.md](https://github.com/eguiraud/correctionlib-gradients/blob/main/ARCHITECTURE.md)
for a broader, higher-level description of the package and its design.

## Floating point precision

JAX aggressively casts to float32.
Maybe in the case of correctionlib we prefer double precision whenever possible?
It can be configured with `from jax import config; config.update("jax_enable_x64", True)`
but it also seems wrong to set it at global scope behind the users' back.
With things as they are now, `test_scalar` would fail with `jit=True` because of loss of precision
if we didn't configure JAX as above at the start of the test.
