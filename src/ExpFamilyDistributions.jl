module ExpFamilyDistributions

using LinearAlgebra
using SpecialFunctions: loggamma, digamma

export ExpFamilyDistribution
export basemeasure
export gradlognorm
export kldiv
export lognorm
export mean
export naturalparam
export stats
export update!


"""
    abstract type ExpFamilyDistribution end

Supertype for distributions member of the exponential family.
"""
abstract type ExpFamilyDistribution end

function loglikelihood(x::AbstractVector)
    Tx = stats(pdf, x)
    η = naturalparam(pdf)
    dot(η, Tx) - lognorm(pdf) + basemeasure(pdf, x)
end

function loglikelihood(x::Number)
    Tx = stats(pdf, x)
    η = naturalparam(pdf)
    η * Tx - lognorm(pdf) + basemeasure(pdf, x)
end

"""
    loglikelihood(pdf, x)

Return the log-likelihood for each of `x` given `pdf`.
"""
loglikelihood


"""
    basemeasure(pdf, x)

Return the base measure of `pdf` for the vector `x`.
"""
basemeasure


"""
    gradlognorm(pdf)

Return the gradient of the log-normalizer w.r.t. the natural
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
    lognorm(pdf)

Return the log-normalization constant of `pdf`
"""
lognorm


"""
    naturalparam(pdf)

Return the natural (a.k.a. the canonical) parameters as a vector.
"""
naturalparam

"""
    stats(pdf, x)

Extract the sufficient statistics of `x` corresponding to type of
`pdf`.
"""
stats

"""
    update!(pdf, η)

Update the parameters given a new natural parameter `η`.
"""
update!

#######################################################################
# Distributions

export AbstractNormal
export Normal
export NormalDiag

include("Normal.jl")

export AbstractGamma
export Gamma
include("Gamma.jl")

export Dirichlet
include("Dirichlet.jl")

end

