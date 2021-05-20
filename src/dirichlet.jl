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
    struct Dirichlet{P<:AbstractParameter,D} <: AbstractDirichlet{D}
        param::P
    end

Dirichlet distribution.

# Constructors

    Dirichlet(α)

where `α` is a vector of concentrations.

# Examples
```jldoctest
julia> Dirichlet([1.0, 2.0, 3.0])
Dirichlet{DefaultParameter{Vector{Float64}}, 3}:
  α = [1.0, 2.0, 3.0]
```
"""
struct Dirichlet{P<:AbstractParameter,D} <: AbstractDirichlet{D}
    param::P
end

function Dirichlet(α::AbstractVector)
    param = DefaultDirichletParameter(α)
    P = typeof(param)
    D = length(α)
    Dirichlet{P,D}(param)
end

#######################################################################
# Distribution interface.

basemeasure(::AbstractDirichlet, x::AbstractVector) = -log.(x)

function lognorm(d::AbstractDirichlet, η::AbstractVector = naturalform(d.param))
    sum(loggamma.(η)) - loggamma(sum(η))
end

function sample(d::AbstractDirichlet, size)
    α = stdparam(d)
    d_ = Dists.Dirichlet(d.α)
    [Dists.rand(d_) for i in 1:size]
end

# vectorization is effect less
splitgrad(d::AbstractDirichlet, μ) = identity(μ)

stats(::AbstractDirichlet, x::AbstractVector) = log.(x)

function stdparam(d::AbstractDirichlet, η::AbstractVector = naturalform(d.param))
    (α = η,)
end

