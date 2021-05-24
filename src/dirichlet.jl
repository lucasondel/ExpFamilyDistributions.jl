# SPDX-License-Identifier: MIT

"""
    AbstractDirichlet{D} <: Distribution

Abstract type for Dirichlet distribution implementations.
"""
abstract type AbstractDirichlet{D} <: Distribution end

#######################################################################
# Parameter of the Dirichlet distribution.

function DefaultDirichletParameter(α)
    DefaultParameter(α)
end

#######################################################################
# Dirichlet distribution

"""
    mutable struct Dirichlet{D} <: AbstractDirichlet{D}
        param::P where P <: AbstractParameter
    end

Dirichlet distribution.

# Constructors

    Dirichlet(α)

where `α` is a vector of concentrations.

# Examples
```jldoctest
julia> Dirichlet([1.0, 2.0, 3.0])
Dirichlet{3}:
  α = [1.0, 2.0, 3.0]
```
"""
mutable struct Dirichlet{D} <: AbstractDirichlet{D}
    param::P where P <: AbstractParameter
end

function Dirichlet(α::AbstractVector)
    param = DefaultDirichletParameter(α)
    P = typeof(param)
    D = length(α)
    Dirichlet{D}(param)
end

#######################################################################
# Distribution interface.

basemeasure(::AbstractDirichlet, x::AbstractVector) = -log.(x)

function lognorm(d::AbstractDirichlet, η::AbstractVector = naturalform(d.param))
    sum(loggamma_dot(η)) - loggamma(sum(η))
end

function gradlognorm(d::AbstractDirichlet)
    α = stdparam(d).α
    digamma.(α) .- digamma(sum(α))
end

function sample(d::AbstractDirichlet{D}, size) where D
    α = stdparam(d).α
    retval = similar(α, D, size)
    d_ = Dists.Dirichlet(Array(α))
    retval[:, :] = Dists.rand(d_, size)
    retval
end

# vectorization is effect less
splitgrad(d::AbstractDirichlet, μ) = μ

stats(::AbstractDirichlet, x::AbstractVector) = log.(x)

function stdparam(d::AbstractDirichlet, η::AbstractVector = naturalform(d.param))
    (α = η,)
end

