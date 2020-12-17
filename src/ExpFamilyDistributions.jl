module ExpFamilyDistributions

using LinearAlgebra
using SpecialFunctions: loggamma, digamma

export ExpFamilyDistribution
export basemeasure
export gradlognorm
export kldiv
export lognorm
export loglikelihood
export mean
export naturalparam
export stats
export update!

"""
    abstract type ExpFamilyDistribution end

Supertype for distributions member of the exponential family.
"""
abstract type ExpFamilyDistribution end

"""
    loglikelihood(p, x)

Returns the log-likelihood of `x` for the distribution `p`.
"""
function loglikelihood(p, x)
    Tx = stats(p, x)
    η = naturalparam(p)
    dot(η, Tx) - lognorm(p) + basemeasure(p, x)
end

"""
    basemeasure(p, x)

Returns the base measure of `x` for the distribution `p`.
"""
basemeasure

"""
    gradlognorm(p)

Returns the gradient of the log-normalizer of `p` w.r.t. its natural
parameters.
"""
gradlognorm

"""
    kldiv(q::T, p::T) where T<:ExpFamilyDistribution

Compute the KL-divergence between two distributions of the same type
(i.e. `kldiv(Normal, Normal)`, `kldiv(Dirichlet, Dirichlet)`, ...)
"""
function kldiv(q::T, p::T) where T<:ExpFamilyDistribution
    q_η, p_η = naturalparam(q), naturalparam(p)
    lognorm(p) - lognorm(q) - dot(p_η .- q_η, gradlognorm(q))
end

"""
    lognorm(p)

Returns the log-normalization constant of `p`.
"""
lognorm

"""
    mean(p)

Returns the mean of the distribution `p`.
"""
mean

"""
    naturalparam(p)

Returns the natural parameters of `p`.
"""
naturalparam

"""
    stats(p, x)

Returns the sufficient statistics of `x` for the distribution `p`.
"""
stats

"""
    update!(p, η)

Updates the parameters given a new natural parameter `η`.
"""
update!

#######################################################################
# Distributions

export AbstractNormal
export Normal
export NormalDiag

include("normal.jl")

export AbstractGamma
export Gamma
include("gamma.jl")

export Dirichlet
include("dirichlet.jl")

end

