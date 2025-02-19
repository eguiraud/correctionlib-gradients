# correctionlib-gradients' architecture

A high-level description of correctionlib-gradients' design.

The goal of this project is to perform autodifferentiation of
[correctionlib](https://github.com/cms-nanoAOD/correctionlib) correction
calculations.

The main idea is to produce a Python function that evaluates a correction by
walking the compute graph described by `correctionlib.schemav2.Correction` (i.e.
a Python version of the JSON representation of a correction).

[JAX](https://jax.readthedocs.io/en/latest/index.html) can then see what happens
to values that pass through this compute graph evaluator and autodifferentiate
accordingly (I had
[a quick discussion](https://github.com/google/jax/discussions/17715#discussioncomment-7077462)
with JAX developers about how to implement this).

We do have to teach JAX how to propagate gradients through some operations, e.g.
bin look-ups, which is done by providing the implementation of the corresponding
Jacobian vector product (see JAX docs).

We then wrap the JAX-friendly compute graph evaluation in a
`CorrectionWithGradient` object that mimics the interface of
`correctionlib.highlevel.Correction`.

## Correction evaluations and JAX tracing

JAX needs to be able to trace computations through a call to
`CorrectionWithGradient.evaluate` in order to support features such as `jax.jit`
or `jax.vmap`. At the time of writing, only the evaluation of `Formula`
corrections is fully JAX traceable.

The problems with other types of corrections:

- scipy's cubic splines, that we use as a differentiable approximation of a 1D
  histogram profile, at some point explicitly cast the input array to
  `np.array`, and JAX cannot trace what happens to the inputs after that point
- in case of a Formula inside a Binning, we first look up which bin in the
  Binning we care about, and then evaluate the corresponding Formula. During
  tracing, the index that would be used for the bin look-up does not have a
  value, it's just a traced array, and of course that breaks

This topic is tracked at
https://github.com/eguiraud/correctionlib-gradients/issues/42.

## Floating point precision

JAX aggressively casts to float32. In the case of correctionlib we might prefer
double precision whenever possible. It can be configured with
`from jax import config; config.update("jax_enable_x64", True)` but it also
seems wrong to set it at global scope behind the users' back. With things as
they are now, `test_scalar` would fail with `jit=True` because of loss of
precision if we didn't configure JAX as above at the start of the test.

## Alternative considered: Python code generation

Instead of implementing a generic compute graph evaluator through which we pass
JAX arrays, we could instead implement a code generator that takes a
correction's compute graph and produces code for a function that evaluates the
graph. That code can then be `eval`'d. At this point the mechanism is the same
as the current implementation: we can ask JAX to differentiate this function.

I think the main difference would be that evaluating the generated function
should be more efficient than walking the compute graph, if the code generator
is implemented well.

However this is more complicated to implement, debugging through code generation
is much more awkward than stepping through a compute graph walk, and in a sense
this kind of code generation is exactly what `jax.jit` does.

## Duplication of functionality w.r.t. correctionlib

Since JAX has to do a forward pass that actually computes the correction's
output in order to compute the corresponding gradients, this package ends up
being a Python-only reimplementation of correctionlib (or at least of a
differentiable subset of the supported correction types).

The duplication is not ideal, but at the time of writing we feel it would be
more complicated to propagate gradients through the C++ correction
implementations of the original correctionlib package. Which brings us to...

## correctionlib auto-differentiation in C++

`correctionlib-gradients`, by design, only serves Python users. That simplifies
development significantly and lets us move quickly as we experiment with
features and find out about roadblocks. In the longer term, it would be nice to
have a single correctionlib AD implementation that serves both C++ and Python
users.

At the time of writing I do not know of a C++ library that can autodifferentiate
a dynamic compute graph evaluation the same way JAX does in this package (and
implementing such a C++ library is a project of a larger scope than what I feel
like tackling at this point). I think the best option for C++ is implementing a
code generator that produces code that evaluates a given correction, then pass
the generated correction evaluation code through
[enzyme](https://enzyme.mit.edu/) or [clad](https://github.com/vgvassilev/clad).
