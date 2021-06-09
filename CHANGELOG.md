# Releases

## 1.0.1
* fixed: the `fromdict` function cannot be used with subtype of
  `DefaultParameter`.

## 1.0.0
* added functions `todict` and `fromdict` to allow easy storing
  of the parameters
* project is now in version 1.0.0 (starting backward compatibility
  updates)

## 0.10.0
* package is compatible with `CuArray` from the CUDA package
* added `reallocate` function to change distribution's parameter type
  (i.e. from Array to CuArray)
* `Gamma` distribution is "vectorized" i.e. it represents the joint
  distribution of independent Gamma distributed variables
* remove Zygote (gradient of lognormalizers are implemented explicitely)
* replace StatsFuns with SpecialFunctions package
* license is MIT again (main reason: CeCILL-B is not OSI approved)
* updated author's email in `Project.toml`

## 0.9.3
* fixed: no type specialization for the splitgrad function to allow
  using AD toolkit such as AutoGrad.jl

## 0.9.2
* replaced ForwardDiff with Zygote for efficiency reason
* added Zygote adjoint for `vec_tril` and its inverse
* specialized the Wishart `gradlognorm` function for optimal performances
* fixed latex equations in the documentation

## 0.9.1
* fixed: typo in calling Diagonal
* added: test for the DefaultParameter struct

## 0.9.0
* refactored parameters API:
    * abstract parameter structure and interface
    * default parameter implemenation
* support julia 1.6 onward
* change LICENSE to CeCILL-B

## 0.8.0

* added an abstract type for each distribution to allow easy extension
  of the pre-defined distributions.
* Wishart natural parameters and sufficient statistics are more compact
  (Symmetric matrices are replaced by a their diagonal and the lower
  triangular parts)
* each distribution can have arbitrary parameterization through the
  `Parameter` object.
* computation are differentiable (with respect to Zygote) to allow use
  of this package with automatic differentiation packages.
* added the function `splitgrad` to replace the `vectorize` parameter
  in the `gradlognorm` function.

## 0.7.0

* changed the parameterization of the Wishart: the precision matrix
  is decomposed into its diagonal and the lower-triangular part of the
  matrix.
* added the `mu` parameter to the `kldiv` function: this parameter
  allows to provide directly the expectation of the sufficient
  statistcs and allows to easily compute the natural gradient of the
  KL digervence.

## 0.6.0

* changed parameterization of the Normal: the sufficient statistics
  are composed of x, the diagonal of xx^T and the lower-triangular
  part of xx^T. This change is to simplify gradient-based inference
  algorithms: with this parameterization, they don't need to ensure
  that the resulting matrix will be symmetric.
* added `vec_tril` and `inv_vec_tril` function to easily extract the
  lower-triangular part of a square matrix.

## 0.5.0

* added `sample` function

## 0.4.0

* added the Wishart distribution
* bugfix: `gradlognorm(gamma, vectorize = false)` now returns a tuple

## 0.3.1

* bugfix: added MIME type to the `Base.show` redifinition, this avoid
  to have chaotic printing of arrays of distributions
* added this CHANGELOG file to the project

## 0.3.0

* added function `stdparam` to be able to convert the natural
  parameters to the standard ones.

## 0.2.0

* Added "delta" distributions

## 0.1.0

* initial release
