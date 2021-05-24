# SPDX-License-Identifier: MIT

"""
    AbstractNormalDiag{D} <: Distribution

Abstract type for Normal distribution (with diagonal covariance matrix)
implementations.
"""
abstract type AbstractNormalDiag{D} <: Distribution end

function DefaultNormalDiagParameter(μ::AbstractVector{T},
                                    v::AbstractVector{T}) where T
    λ = 1 ./ v
    ξ = vcat(λ .* μ, -T(.5)*λ)
    DefaultParameter(ξ)
end

"""
    mutable struct NormalDiag{D} <: AbstractNormalDiag{D}
        param::P where P <: AbstractParameter
    end

Normal distribution with a diagonal covariance matrix.

# Constructors

    NormalDiag(μ, v)

where `μ` is the mean `v` is the diagonal of the covariance matrix.

# Examples
```jldoctest
julia> NormalDiag([1.0, 1.0], [2.0, 1.0])
NormalDiag{2}:
  μ = [1.0, 1.0]
  Σ = [2.0 0.0; 0.0 1.0]
```
"""
mutable struct NormalDiag{D} <: AbstractNormalDiag{D}
    param::P where P <: AbstractParameter
end

function NormalDiag(μ::AbstractVector{T}, v::AbstractVector{T}) where T
    param = DefaultNormalDiagParameter(μ, v)
    P = typeof(param)
    D = length(μ)
    NormalDiag{D}(param)
end
NormalDiag(μ::AbstractVector, Σ::Diagonal) = NormalDiag(μ, diag(Σ))

#######################################################################
# Distribution interface.

function basemeasure(::AbstractNormalDiag, x::AbstractVector{T}) where T
    -T(.5) * length(x) * log(T(2π))
end

function lognorm(n::AbstractNormalDiag{D},
                 η::AbstractVector{T} = naturalform(n.param)) where {T,D}
    η₁, η₂ = η[1:D], η[D+1:end]
    -T(.5) * sum(log.(-T(2)*η₂)) -T(.25) * dot(η₁, (1 ./ η₂) .* η₁)
end

function gradlognorm(n::AbstractNormalDiag{D}, η = naturalform(n.param)) where D
    T = eltype(η)
    Λμ, nhλ = η[1:D], η[D+1:end]
    v = 1 ./ (-T(2) * nhλ)
    μ = v .* Λμ
    m = v .+ μ.^2
    vcat(μ, m)
end

function sample(n::AbstractNormalDiag{D}, size) where D
    μ, Σ = stdparam(n)
    σ = sqrt.(diag(Σ))
    ϵ = randn!(similar(μ, D, size))
    μ .+ σ .* ϵ
end

stats(::AbstractNormalDiag, x) = vcat(x, x.^2)

_splitgrad_normaldiag(D, m) = m[1:D], m[D+1:end]
splitgrad(n::AbstractNormalDiag{D}, m) where D = m[1:D], m[D+1:end]

function stdparam(n::AbstractNormalDiag{D},
                  η::AbstractVector{T} = naturalform(n.param)) where {T,D}
    Λμ, nhλ = η[1:D], η[D+1:end]
    v = 1 ./ (-T(2) * nhλ)
    μ = v .* Λμ
    (μ = μ, Σ = Diagonal(v))
end

