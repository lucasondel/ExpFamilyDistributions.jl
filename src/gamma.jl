# SPDX-License-Identifier: MIT

"""
    AbstractGamma{D} <: Distribution

Abstract type for Gamma distribution implementations.
"""
abstract type AbstractGamma{D} <: Distribution end

function DefaultGammaParameter(α, β)
    DefaultParameter(vcat(-β, α))
end

"""
    mutable struct Gamma{D} <: AbstractGamma
        param::P where P <: AbstractParameter
    end

Set of `D` independent Gamma distributions.

# Constructors

    Gamma(α, β)

where `α` and `β` are the shape and rate parameters of the
distribution.

# Examples
```jldoctest
julia> Gamma([1.0, 1.0], [2.0, 2.0])
Gamma{2}:
  α = [1.0, 1.0]
  β = [2.0, 2.0]
```
"""
mutable struct Gamma{D} <: AbstractGamma{D}
    param::P where P <: AbstractParameter
end

function Gamma(α::AbstractVector{T}, β::AbstractVector{T}) where T
    D = length(α)
    param = DefaultGammaParameter(α, β)
    Gamma{D}(param)
end

#######################################################################
# Distribution interface

basemeasure(::AbstractGamma, x) = -sum(log, x)

function lognorm(g::AbstractGamma{D}, η = naturalform(g.param)) where D
    η₁, η₂ = η[1:D], η[D+1:end]
    sum(loggamma_dot(η₂) - η₂ .* log.(-η₁))
end

function gradlognorm(g::AbstractGamma)
    α, β = stdparam(g, naturalform(g.param))
    vcat(α ./ β, digamma.(α) .- log.(β))
end

function sample(g::AbstractGamma{D}, size) where D
    retval = similar(g.α, D, size)
    α, β = Array(g.α), Array(g.β)
    for i in 1:length(α)
        g_ = Dists.Gamma(α[i], 1/β[i])
        retval[i, :] = Dists.rand(g_, size)
    end
    retval
end

splitgrad(g::AbstractGamma{D}, μ) where D = μ[1:D], μ[D+1:end]

stats(::Gamma, x) = vcat(x, log.(x))

function stdparam(g::AbstractGamma{D},
                  η::AbstractVector{T} = naturalform(g.param)) where {T,D}
    α = η[D+1:end]
    β = -η[1:D]
    (α = α, β = β)
end

