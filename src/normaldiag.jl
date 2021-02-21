
#######################################################################
# Parameter of the Normal distribution with diagonal covariance matrix.


function DefaultNormalDiagParameter(μ::AbstractVector{T},
                                    v::AbstractVector{T}) where T
    λ = 1 ./ v
    ξ = vcat(λ .* μ, -T(.5)*λ)
    Parameter{T}(ξ, identity, identity)
end

#######################################################################
# Normal distribution with diagonal covariance matrix

"""
    struct NormalDiag{D} <: Distribution
        param::Parameter{T}
    end

Normal distribution with a diagonal covariance matrix.

# Constructors

    NormalDiag{D}(T=Float64)
    NormalDiag(μ[, v])

where `T` is the encoding type of the parameters, `D` is the
dimension of the support, `μ` is a vector and `v` is the diagonal of
the covariance matrix.

# Examples
```jldoctest
julia> NormalDiag{2}(Float32)
NormalDiag{2}:
  μ = Float32[0.0, 0.0]
  Σ = Float32[1.0 0.0; 0.0 1.0]

julia> NormalDiag([1.0, 1.0])
NormalDiag{2}:
  μ = [1.0, 1.0]
  Σ = [1.0 0.0; 0.0 1.0]

julia> NormalDiag([1.0, 1.0], [2.0, 1.0])
NormalDiag{2}:
  μ = [1.0, 1.0]
  Σ = [2.0 0.0; 0.0 1.0]
```
"""
struct NormalDiag{D} <: Distribution
    param::Parameter{T} where T
end

function NormalDiag(μ::AbstractVector{T}, v::AbstractVector{T}) where T
    NormalDiag{length(μ)}(DefaultNormalDiagParameter(μ, v))
end
NormalDiag(μ::AbstractVector) = NormalDiag(μ, ones(eltype(μ), length(μ)))
NormalDiag(μ::AbstractVector, Σ::Diagonal) = NormalDiag(μ, diag(Σ))
NormalDiag{D}(T::Type = Float64) where D = NormalDiag(zeros(T, D), ones(T, D))

#######################################################################
# Distribution interface.

function basemeasure(::NormalDiag{D}, x::AbstractVector{T}) where {T,D}
    length(x) == D || throw(DimensionMismatch("expected input dimension $D got $(length(x))"))
    -T(.5) * length(x) * log(T(2π))
end

function lognorm(n::NormalDiag{D},
                 η::AbstractVector{T} = naturalform(n.param)) where {T,D}
    η₁, η₂ = η[1:D], η[D+1:end]
    -T(.5) * sum(log.(-T(2)*η₂)) -T(.25) * dot(η₁, (1 ./ η₂) .* η₁)
end

function sample(n::NormalDiag{D}, size) where D
    μ, Σ = stdparam(n)
    T = eltype(μ)
    σ = sqrt.(diag(Σ))
    [μ + σ .* randn(T, D) for i in 1:size]
end

stats(::NormalDiag, x) = vcat(x, x.^2)

_splitgrad_normaldiag(D, m) = m[1:D], m[D+1:end]
splitgrad(n::NormalDiag{D}, m) where D = m[1:D], m[D+1:end]

function stdparam(n::NormalDiag{D},
                  η::AbstractVector{T} = naturalform(n.param)) where {T,D}
    Λμ, nhλ = η[1:D], η[D+1:end]
    v = 1 ./ (-T(2) * nhλ)
    μ = v .* Λμ
    (μ = μ, Σ = Diagonal(v))
end

