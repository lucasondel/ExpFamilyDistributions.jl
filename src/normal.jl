# SPDX-License-Identifier: MIT

"""
    AbstractNormal{D} <: Distribution

Abstract type for Normal distribution implementations.
"""
abstract type AbstractNormal{D} <: Distribution end

function DefaultNormalParameter(μ::AbstractVector{T},
                                Σ::AbstractMatrix{T}) where T
    Λ = pdmat_inverse(Σ)
    ξ = vcat(Λ*μ, -T(.5)*diag(Λ), -vec_tril(Λ))
    DefaultParameter(ξ)
end

"""
    mutable struct Normal{D} <: AbstractNormal{D}
        param::P where P <: AbstractParameter
    end

Normal distribution with full covariance matrix.

# Constructors

Normal(μ, Σ)

where `μ` is the mean and `Σ` is the covariance matrix.

# Examples
```jldoctest
julia> Normal([1.0, 1.0], [2.0 0.5; 0.5 1.0])
Normal{2}:
  μ = [0.9999999999999998, 1.0]
  Σ = [2.0 0.5; 0.5 1.0]
```
"""
mutable struct Normal{D} <: AbstractNormal{D}
    param::P where P <: AbstractParameter
end

function Normal(μ, Σ)
    param = DefaultNormalParameter(μ, Σ)
    P = typeof(param)
    D = length(μ)
    Normal{D}(param)
end

_unpack(D, v) = v[1:D], v[D+1:2*D], v[2*D+1:end]

#######################################################################
# Distribution interface.

function basemeasure(::AbstractNormal, x::AbstractVector{T}) where T
    -T(.5) * length(x) * log(T(2π))
end

function lognorm(n::AbstractNormal{D},
                 η::AbstractVector{T} = naturalform(n.param)) where {T,D}
    η₁, η₂, η₃ = _unpack(D, η)
    H₂ = matrix(η₂, T(.5)*η₃)
    -T(.5)*pdmat_logdet(-T(2)*H₂) + T(.25)*dot(η₁, pdmat_inverse(-H₂) * η₁)
end

function gradlognorm(n::AbstractNormal)
    μ, Σ = stdparam(n, naturalform(n.param))
    M = Σ + μ*μ'
    vcat(μ, diag(M), vec_tril(M))
end

function sample(n::AbstractNormal{D}, size) where D
    μ, Σ = stdparam(n)
    L = cholesky(Σ).L
    ϵ = randn!(similar(μ, D, size))
    μ .+ L*ϵ
end

splitgrad(n::AbstractNormal{D}, m) where D = _unpack(D, m)

function stats(::AbstractNormal{D}, x::AbstractVector) where D
    xxᵀ = x * x'
    vcat(x, diag(xxᵀ), vec_tril(xxᵀ))
end

function stdparam(n::AbstractNormal{D},
                  η::AbstractVector{T} = naturalform(n.param)) where {T,D}
    η₁, η₂, η₃ = _unpack(D, η)
    H₂ = matrix(η₂, T(.5)*η₃)
    Σ = pdmat_inverse(-T(2)*H₂)
    (μ = Σ*η₁, Σ = Σ)
end

