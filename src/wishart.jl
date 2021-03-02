
#######################################################################
# Super-type

"""
    AbstractWishart{D} <: Distribution

Abstract type for Wishart distribution implementations.
"""
abstract type AbstractWishart{D} <: Distribution end

#######################################################################
# Parameter of the Gamma distribution.

function DefaultWishartParameter(W, v)
    T = eltype(W)
    M = inv(W)
    η₁ = -T(.5)*diag(M)
    η₂ = -vec_tril(M)
    η₃ = T(.5)*v
    Parameter{T}(vcat(η₁, η₂, η₃), identity, identity)
end

#######################################################################
# Wishart distribution

"""
    struct Wishart{D} <: AbstractWishart{D}
        param
    end

Wishart distribution.

# Constructors

    Wishart{D}()
    Wishart(W[, v])

where `T` is the encoding type of the parameters and `W` is a
positive definite DxD matrix.

# Examples
```jldoctest
julia> Wishart{2}(Float32)
Wishart{2}:
  W = Float32[1.0 0.0; 0.0 1.0]
  v = 2.0

julia> Wishart([1 0.5; 0.5 1])
Wishart{2}:
  W = [1.0 0.5; 0.5 1.0]
  v = 2.0
```
"""
struct Wishart{D} <: AbstractWishart{D}
    param::Parameter{T} where T


end

function Wishart(W::AbstractMatrix{T}, v::Real) where T<:Real
    Wishart{size(W, 1)}(DefaultWishartParameter(W, v))
end

function Wishart(W::AbstractMatrix, v)
    T = eltype(W)
    Wishart(W, T(v))
end
Wishart(W::AbstractMatrix) = Wishart(W, size(W,1))

function Wishart{D}(T::Type = Float64) where D
    W = Symmetric(Matrix{T}(I,D,D))
    v = T(D)
    Wishart(W, v)
end

#######################################################################
# Distribution interface

function basemeasure(w::Wishart, X::Symmetric)
    D = size(X, 1)
    -.5*(D-1)*logdet(X) - .25*D*(D-1)log(pi)
end

function lognorm(w::Wishart{D},
                 η::AbstractVector{T} = naturalform(w.param)) where {T,D}
    diagM = η[1:D]
    trilM = η[D+1:end-1]
    M = -T(2)*matrix(diagM, T(.5)*trilM)
    v = T(2)*η[end]
    #W = inv(-T(2)*M)
    #v = T(2)*η[end]
    #(W = W, v = v)

    retval = T(0.5)*(-v*logdet(M) + v*D*T(log(2)))
    retval += sum([loggamma(T(0.5)*(v+1-i)) for i in 1:D])

    #W, v = stdparam(w)
    #T(.5)*(v*logdet(W) +  v*D*T(log(2)) ) + sum([loggamma(T((v+1-i)/2)) for i in 1:D])
end

stats(w::Wishart, X::AbstractMatrix) = vcat(diag(X), vec_tril(X), logdet(X))

function sample(w::Wishart, size = 1)
    w_ = Dists.Wishart(w.v, PDMat(w.W))
    [rand(w_) for i in 1:size]
end

function splitgrad(w::Wishart{D}, μ::AbstractVector{T}) where {T,D}
    diag_∂₁ = μ[1:D]
    tril_∂₁ = μ[D+1:end-1]
    #∂₁ = matrix(diag_∂₁, T(.5)*tril_∂₁)
    diag_∂₁, tril_∂₁, μ[end]
end

function stdparam(w::Wishart{D},
                  η::AbstractVector{T} = naturalform(w.param)) where {T,D}
    diagM = η[1:D]
    trilM = η[D+1:end-1]
    M = matrix(diagM, T(.5)*trilM)
    W = inv(-T(2)*M)
    v = T(2)*η[end]
    (W = W, v = v)
end
