
#######################################################################
# Super-type

"""
    AbstractNormal{D} <: Distribution

Abstract type for Normal distribution implementations.
"""
abstract type AbstractNormal{D} <: Distribution end

#######################################################################
# Normal distribution parameters.


function DefaultNormalParameter(μ::AbstractVector{T},
                                Σ::AbstractMatrix{T}) where T
    Λ = inv(Σ)
    ξ = vcat(Λ*μ, -T(.5)*diag(Λ), -vec_tril(Λ))
    Parameter(ξ, identity, identity)
end

#######################################################################
# Normal distribution with full covariance matrix

"""
    struct Normal{P<:Parameter,D} <: AbstractNormal{D}
        param::P
    end

Normal distribution with full covariance matrix.

# Constructors

Normal(μ, Σ)

where `μ` is the mean and `Σ` is the covariance matrix.

# Examples
```jldoctest
julia> Normal([1.0, 1.0], [2.0 0.5; 0.5 1.0])
Normal{Parameter{Array{Float64,1}},2}:
  μ = [1.0, 1.0]
  Σ = [2.0 0.5; 0.5 1.0]
```
"""
struct Normal{P<:Parameter,D} <: AbstractNormal{D}
    param::P
end

function Normal(μ, Σ)
    param = DefaultNormalParameter(μ, Σ)
    P = typeof(param)
    D = length(μ)
    Normal{P,D}(param)
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
    -.5*logdet(-T(2)*H₂) - T(.25)*dot(η₁, inv(H₂), η₁)
end

function sample(n::AbstractNormal{D}, size) where D
    μ, Σ = stdparam(n)
    T = eltype(μ)
    L = cholesky(Σ).L
    [μ + L*randn(T, D) for i in 1:size]
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
    Σ = inv(-T(2)*H₂)
    (μ = Σ*η₁, Σ = Σ)
end

