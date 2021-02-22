
#######################################################################
# Normal distribution parameters.


function DefaultNormalParameter(μ::AbstractVector{T},
                                Σ::AbstractMatrix{T}) where T
    Λ = inv(Symmetric(Σ))
    ξ = vcat(Λ*μ, -T(.5)*diag(Λ), -vec_tril(Λ))
    Parameter{T}(ξ, identity, identity)
end

#######################################################################
# Normal distribution with full covariance matrix

"""
    struct Normal{D} <: Distribution
        param::Parameter{T} where T
    end

Normal distribution with full covariance matrix.

# Constructors

Normal{D}(T = Float64)
    Normal(μ[, Σ])

where `T` is the encoding type of the parameters, `D` is the
dimension of the support, `μ` is a vector and `Σ` is a
positive-definite matrix.

# Examples
```jldoctest
julia> Normal{2}(Float32)
Normal{2}:
  μ = Float32[0.0, 0.0]
  Σ = Float32[1.0 0.0; 0.0 1.0]

julia> Normal([1.0, 1.0])
Normal{2}:
  μ = [1.0, 1.0]
  Σ = [1.0 0.0; 0.0 1.0]

julia> Normal([1.0, 1.0], [2.0 0.5; 0.5 1.0])
Normal{2}:
  μ = [1.0, 1.0]
  Σ = [2.0 0.5; 0.5 1.0]
```
"""
struct Normal{D} <: Distribution
    param::Parameter{T} where T
end

function Normal(μ, Σ) where T
    Normal{length(μ)}(DefaultNormalParameter(μ, Σ))
end

function Normal{D}(T::Type = Float64) where {D}
    Normal(zeros(T, D), Matrix{T}(I, D, D))
end

function Normal(μ::AbstractVector)
    T = eltype(μ)
    D = length(μ)
    Normal(μ, Matrix{T}(I, D, D))
end

_unpack(D, v) = v[1:D], v[D+1:2*D], v[2*D+1:end]

#######################################################################
# Distribution interface.

function basemeasure(::Normal{D}, x::AbstractVector{T}) where{T,D}
    length(x) == D || throw(DimensionMismatch("expected input dimension $D got $(length(x))"))
    -T(.5) * length(x) * log(T(2π))
end

function lognorm(n::Normal{D},
                 η::AbstractVector{T} = naturalform(n.param)) where {T,D}
    η₁, η₂, η₃ = _unpack(D, η)
    H₂ = matrix(η₂, T(.5)*η₃)
    -.5*logdet(-T(2)*H₂) - T(.25)*dot(η₁, inv(H₂), η₁)
end

function sample(n::Normal{D}, size) where D
    μ, Σ = stdparam(n)
    T = eltype(μ)
    L = cholesky(Σ).L
    [μ + L*randn(T, D) for i in 1:size]
end

splitgrad(n::Normal{D}, m) where D = _unpack(D, m)

function stats(::Normal{D}, x::AbstractVector) where D
    length(x) == D || throw(DimensionMismatch("expected input dimension $D got $(length(x))"))
    xxᵀ = x * x'
    vcat(x, diag(xxᵀ), vec_tril(xxᵀ))
end

function stdparam(n::Normal{D}, η::AbstractVector{T} = naturalform(n.param)) where {T,D}
    η₁, η₂, η₃ = _unpack(D, η)
    H₂ = matrix(η₂, T(.5)*η₃)
    Σ = inv(-T(2)*H₂)
    (μ = Σ*η₁, Σ = Σ)
end

