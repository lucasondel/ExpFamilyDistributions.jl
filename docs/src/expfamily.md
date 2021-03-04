# Exponential Family Distributions

## Exponential Family
All the distributions provided by this package are members of the
exponential family of distribution, i.e. they have the follotwing
canonical form:
```math
p(x) = \exp \{ \eta^\top T(x) - A(\eta) + B(x) \}
```
where:
* ``\eta`` is the natural parameter (scalar or vector)
* ``T(x)`` is the sufficient statistic (scalar or vector)
* ``A(\eta)`` is the log-normalizer (scalar)
* ``B(x)`` is the base measure (scalar)

Practically, the package provide the following abstract type
```@docs
Distribution
```
Which represents the supter-type of members of the exponential family.

## Parameterization

All subtypes of [`Distribution`](@ref) have the following form:
```julia
struct MyDistribution{P<:AbstractParam} <: Distribution
    param::P
end
```
This particular form allows each distribution to be agnostic to their
concrete parameterization. The parameter type inherits from:
```@docs
AbstractParameter
```
and supports the following methods
```@docs
naturalform
realform
jacobian
```

## Distribution interface

Each subtype of [`Distribution`] implements the following interface:
* [`basemeasure`](@ref)
* [`gradlognorm`](@ref)
* [`kldiv`](@ref)
* [`loglikelihood`](@ref)
* [`lognorm`](@ref)
* [`sample`](@ref)
* [`splitgrad`](@ref)
* [`stats`](@ref)
* [`stdparam`](@ref)

```@docs
basemeasure
gradlognorm
kldiv
lognorm
loglikelihood
sample
splitgrad
stats
stdparam
```

## Utilities

The package also provides the following utility functions:
```@docs
vec_tril
inv_vec_tril
matrix
```

