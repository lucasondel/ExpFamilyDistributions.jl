
#######################################################################
# Super-type

"""
    AbstractNormalDiag{D} <: Distribution

Abstract type for Normal distribution (with diagonal covariance matrix)
implementations.
"""
abstract type AbstractNormalDiag{D} <: Distribution end

#######################################################################
# Parameter of the Normal distribution with diagonal covariance matrix.


function DefaultNormalDiagParameter(μ::AbstractVector{T},
                                    v::AbstractVector{T}) where T
    λ = 1 ./ v
    ξ = vcat(λ .* μ, -T(.5)*λ)
    Parameter(ξ, identity, identity)
end

#######################################################################
# Normal distribution with diagonal covariance matrix

"""
    struct NormalDiag{P<:Parameter,D} <: AbstractNormalDiag{D}
        param::P
    end

Normal distribution with a diagonal covariance matrix.

# Constructors

    NormalDiag(μ, v)

where `μ` is the mean `v` is the diagonal of the covariance matrix.

# Examples
```jldoctest
julia> NormalDiag([1.0, 1.0], [2.0, 1.0])
NormalDiag{Parameter{Array{Float64,1}},2}:
  μ = [1.0, 1.0]
  Σ = [2.0 0.0; 0.0 1.0]
```
"""
struct NormalDiag{P<:Parameter,D} <: AbstractNormalDiag{D}
    param::P
end

function NormalDiag(μ::AbstractVector{T}, v::AbstractVector{T}) where T
    param = DefaultNormalDiagParameter(μ, v)
    P = typeof(param)
    D = length(μ)
    NormalDiag{P,D}(param)
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

function sample(n::AbstractNormalDiag{D}, size) where D
    μ, Σ = stdparam(n)
    T = eltype(μ)
    σ = sqrt.(diag(Σ))
    [μ + σ .* randn(T, D) for i in 1:size]
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

