# correctionlib-gradients' architecture

A high-level description of correctionlib-gradients' design.

The goal of this project is to perform autodifferentiation of
[correctionlib](https://github.com/cms-nanoAOD/correctionlib) correction calculations.

The main idea is to produce a Python function that evaluates a correction
by walking the compute graph described by `correctionlib.schemav2.Correction` (basically a
Python version of the JSON representation of a correction).
This is `_base.apply_ast(ast, correction_name, inputs)`. From that generic function we
obtain a function that evaluates a specific correction with `partial(apply_ast(ast, name))`.

[JAX](https://jax.readthedocs.io/en/latest/index.html) can then see what happens to values
that pass through this compute graph evaluator and autodifferentiate accordingly (I had
[a quick discussion](https://github.com/google/jax/discussions/17715#discussioncomment-7077462) with
JAX developers about how to implement this).

We do have to teach JAX how to propagate gradients through some operations, e.g. bin look-ups,
which is done by providing the implementation of the corresponding Jacobian vector product
(see JAX docs).

We then wrap the JAX-friendly compute graph evaluation in a `CorrectionWithGradient` object
that mimics the interface of `correctionlib.highlevel.Correction` but evaluates the correction
as well as its gradient. `jax.grad` is implemented in terms of `jax.value_and_grad` so there is
really no performance difference to retrieve value and gradient both from JAX.

I said "mimics the interface" but that's not quite true: the signature of `evaluate` is different.
`correctionlib.Correction.evaluate` takes positional (nameless) parameters, one per input
variable to the correction, and expects that users pass those values in the correct order.
`CorrectionWithGradient.evaluate` takes a dictionary of inputs, with keys corresponding to the
names of the input variables. This fits well with the extra parameter `wrt` that lets users
specify with respect to which variables gradients should be computed, selecting inputs by name.
In the future, `correctionlib` might also add an `evaluate` signature that takes inputs by name,
as per [this discussion](https://github.com/cms-nanoAOD/correctionlib/issues/166).

## Alternative considered: Python code generation

Instead of implementing a generic compute graph evaluator through which we pass
JAX arrays, we could instead implement a code generator that takes a correction's
compute graph and produces code for a function that evaluates the graph.
That code can then be `eval`'d. At this point the mechanism is the same as the
current implementation: we can ask JAX to differentiate this function.

I think the main difference would be that evaluating the generated function should
be more efficient than walking the compute graph, if the code generator is implemented
well.

However this is more complicated to implement, debugging through code generation is
much more awkward than stepping through a compute graph walk, and in a sense this
kind of code generation is exactly what `jax.jit` does.

## Duplication of functionality w.r.t. correctionlib

Since JAX has to do a forward pass that actually computes the correction's output in
order to compute the corresponding gradients, this package ends up being a Python-only
reimplementation of correctionlib (or at least of a differentiable subset of the supported
correction types).

The duplication is not ideal, but at the time of writing we feel it would
be more complicated to propagate gradients through the C++ correction implementations of the
original correctionlib package. Which brings us to...

## correctionlib autodifferentiation in C++

`correctionlib-gradients`, by design, only serves Python users.
That simplifies development significantly and lets us move quickly as we experiment
with features and find out about roadblocks.
In the longer term, it would be nice to have a single correctionlib AD implementation
that serves both C++ and Python users.

At the time of writing I do not know of a C++ library that can autodifferentiate the
compute graph evaluation the same way JAX does in this package.
And implementing such a C++ library is a project of a larger scope than we feel like
tackling at this point. We could check where things break exactly with an
existing AD library such as [enzyme](https://enzyme.mit.edu).

Another option to explore is to perform C++ code generation of a function that evaluates
the compute graph and then pass that function through [clad](https://github.com/vgvassilev/clad).
