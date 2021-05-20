# SPDX-License-Identifier: MIT

"""
    AbstractGamma <: Distribution

Abstract type for Gamma distribution implementations.
"""
abstract type AbstractGamma <: Distribution end

function DefaultGammaParameter(α, β)
    DefaultParameter(vcat(-β, α))
end

"""
    struct Gamma{P<:AbstractParameter} <: AbstractGamma
        param::P
    end

Gamma distribution.

# Constructors

    Gamma(α, β)

where `α` and `β` are the shape and reate parameters of the
distribution.

# Examples
```jldoctest
julia> Gamma(1, 2)
Gamma{DefaultParameter{Vector{Float64}}}:
  α = 1.0
  β = 2.0
```
"""
mutable struct Gamma{P<:AbstractParameter} <: AbstractGamma
    param::P
end

Gamma(α::T, β::T) where T<:AbstractFloat = Gamma(DefaultGammaParameter(α, β))
Gamma(α::Real, β::Real) = Gamma(Float64(α), Float64(β))

#######################################################################
# Distribution interface

basemeasure(::AbstractGamma, x) = -log(x)

function lognorm(g::AbstractGamma, η::AbstractVector = naturalform(g.param))
    η₁, η₂ = η
    #loggamma(g.α) - g.α * log.(g.β)
    loggamma(η₂) - η₂ * log(-η₁)
end

function sample(g::AbstractGamma, size)
    g_ = Dists.Gamma(g.α, 1/g.β)
    [Dists.rand(g_) for i in 1:size]
end

splitgrad(g::AbstractGamma, μ) = μ[1], μ[2]

stats(::Gamma, x) = [x, log(x)]

function stdparam(g::AbstractGamma,
                  η::AbstractVector{T} = naturalform(g.param)) where T
    (α = η[2], β = -η[1])
end

