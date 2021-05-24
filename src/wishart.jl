# SPDX-License-Identifier: MIT

"""
    AbstractWishart{D} <: Distribution

Abstract type for Wishart distribution implementations.
"""
abstract type AbstractWishart{D} <: Distribution end

function DefaultWishartParameter(W, v)
    T = eltype(W)
    M = pdmat_inverse(W)
    η₁ = -T(.5)*diag(M)
    η₂ = -vec_tril(M)
    η₃ = T(.5)*v
    DefaultParameter(vcat(η₁, η₂, η₃))
end

"""
    mutable struct Wishart{D} <: AbstractWishart{D}
        param::P where P <: AbstractParameter
    end

Wishart distribution.

# Constructors

    Wishart{D}()
    Wishart(W[, v])

where `T` is the encoding type of the parameters and `W` is a
positive definite DxD matrix.

# Examples
```jldoctest
julia> Wishart([1 0.5; 0.5 1], 2)
Wishart{2}:
  W = [0.9999999999999997 0.5; 0.5 1.0]
  v = 2.0
```
"""
mutable struct Wishart{D} <: AbstractWishart{D}
    param::P where P <: AbstractParameter
end

function Wishart(W::AbstractMatrix, v)
    param = DefaultWishartParameter(W, v)
    P = typeof(param)
    D = size(W,1)
    Wishart{D}(param)
end

#######################################################################
# Distribution interface

function basemeasure(w::AbstractWishart, X::AbstractMatrix)
    D = size(X, 1)
    -.5*(D-1)*pdmat_logdet(X) - .25*D*(D-1)log(pi)
end

function gradlognorm(w::AbstractWishart)
    W, v = stdparam(w, naturalform(w.param))
    D = size(W, 1)
    T = eltype(W)
    ∂η₁ = v * W
    ∂η₂ = sum([digamma((T(v+1-i)/2)) for i in 1:D]) + T(D*log(2)) + pdmat_logdet(W)
    vcat(diag(∂η₁), vec_tril(∂η₁), ∂η₂)
end

function lognorm(w::AbstractWishart{D},
                 η::AbstractVector{T} = naturalform(w.param)) where {T,D}
    diagM = η[1:D]
    trilM = η[D+1:end-1]
    M = -T(2)*matrix(diagM, T(.5)*trilM)
    v = T(2)*sum(η[end:end]) # sum of 1-element array to run to bypass
                             # scalar operation issue

    retval = T(0.5)*(-v * pdmat_logdet(M) + v*D*T(log(2)))
    retval += sum([loggamma(T(0.5) * (v+1- i)) for i in 1:D])
end

stats(w::AbstractWishart, X::AbstractMatrix) = vcat(diag(X), vec_tril(X), pdmat_logdet(X))

function sample(w::AbstractWishart{D}, size = 1) where D
    retval = similar(naturalform(w.param), D, D, size)
    w_ = Dists.Wishart(w.v, PDMat(Array(w.W)))
    retval[:,:,:] = cat(rand(w_, size)..., dims = 3)
    retval
end

function splitgrad(w::AbstractWishart{D}, μ) where {T,D}
    diag_∂₁ = μ[1:D]
    tril_∂₁ = μ[D+1:end-1]
    diag_∂₁, tril_∂₁, sum(μ[end:end])
end

function stdparam(w::AbstractWishart{D},
                  η::AbstractVector{T} = naturalform(w.param)) where {T,D}
    diagM = η[1:D]
    trilM = η[D+1:end-1]
    M = matrix(diagM, T(.5)*trilM)
    W = pdmat_inverse(-T(2)*M)
    v = T(2)*sum(η[end:end])
    (W = W, v = v)
end

