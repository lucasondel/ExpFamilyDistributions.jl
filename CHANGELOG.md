# Releases

## 0.8.0

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
