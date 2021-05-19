# SPDX-License-Identifier: MIT

"""
    AbstractNormal{D} <: Distribution

Abstract type for Normal distribution implementations.
"""
abstract type AbstractNormal{D} <: Distribution end

function DefaultNormalParameter(μ::AbstractVector{T},
                                Σ::AbstractMatrix{T}) where T
    Λ = inv(Σ)
    ξ = vcat(Λ*μ, -T(.5)*vec(Λ))
    DefaultParameter(ξ)
end

"""
    struct Normal{P<:AbstractParameter,D} <: AbstractNormal{D}
        param::P
    end

Normal distribution with full covariance matrix.

# Constructors

Normal(μ, Σ)

where `μ` is the mean and `Σ` is the covariance matrix.

# Examples
```jldoctest
julia> Normal([1.0, 1.0], [2.0 0.5; 0.5 1.0])
Normal{DefaultParameter{Vector{Float64}}, 2}:
  μ = [1.0, 1.0]
  Σ = [2.0 0.5; 0.5 1.0]
```
"""
struct Normal{P<:AbstractParameter,D} <: AbstractNormal{D}
    param::P
end

function Normal(μ, Σ)
    param = DefaultNormalParameter(μ, Σ)
    P = typeof(param)
    D = length(μ)
    Normal{P,D}(param)
end

_unpack(D, v) = v[1:D], Hermitian(reshape(v[D+1:end], D, D))

#######################################################################
# Distribution interface.

function basemeasure(::AbstractNormal, x::AbstractVector{T}) where T
    -T(.5) * length(x) * log(T(2π))
end

function gradlognorm(n::AbstractNormal)
    μ, Σ = stdparam(n, naturalform(n.param))
    vcat(μ, vec(Σ + μ*μ'))
end

function lognorm(n::AbstractNormal{D},
                 η::AbstractVector{T} = naturalform(n.param)) where {T,D}
    η₁, H₂ = _unpack(D, η)
    -T(.5)*logdet(-T(2)*H₂) - T(.25)*dot(η₁, inv(H₂), η₁)
end

function sample(n::AbstractNormal{D}, size) where D
    μ, Σ = stdparam(n)
    T = eltype(μ)
    L = cholesky(Σ).L
    [μ + L*randn(T, D) for i in 1:size]
end

splitgrad(n::AbstractNormal{D}, m) where D = _unpack(D, m)

stats(::AbstractNormal{D}, x::AbstractVector) where D = vcat(x, vec(x*x'))

function stdparam(n::AbstractNormal{D},
                  η::AbstractVector{T} = naturalform(n.param)) where {T,D}
    η₁, H₂ = _unpack(D, η)
    Σ = inv(-T(2)*H₂)
    (μ = Σ*η₁, Σ = Σ)
end

