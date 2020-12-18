module ExpFamilyDistributions

using LinearAlgebra
using SpecialFunctions: loggamma, digamma

export ExpFamilyDistribution
export δDistribution

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
# δ-Distributions

"""
    abstract type δDistribution

Supertype for the δ-distributions.
"""
abstract type δDistribution end

function Base.show(io::IO, d::δDistribution)
    println(io, typeof(d), ":")
    print(io, "  μ = ", d.μ)
end

mean(d::δDistribution) = d.μ

#######################################################################
# Distributions

export Normal
export δNormal
export NormalDiag
export δNormalDiag

include("normal.jl")

export Gamma
export δGamma

include("gamma.jl")

export Dirichlet
export δDirichlet

include("dirichlet.jl")

end

