module ExpFamilyDistributions

using LinearAlgebra

export ExpFamilyDistribution
export basemeasure
export gradlognorm
export kldiv
export lognorm
export mean
export naturalparam
export stats
export stdparam
export update!


"""
    abstract type ExpFamilyDistribution end

Supertype for distributions member of the exponential family.
"""
abstract type ExpFamilyDistribution end


"""
    (pdf::ExpFamilyDistribution)(X::Matrix{<:AbstractFloat})

Return the log-likelihood for each columne of `X`.
"""
function (pdf::ExpFamilyDistribution)(X::Matrix{<:AbstractFloat})
    TX = stats(pdf, X)
    η = naturalparam(pdf)
    TX' * η .- lognorm(pdf) .+ basemeasure(pdf, X)
end


"""
    basemeasure(pdf::ExpFamilyDistribution, x)

Return the base measure of `pdf` for the vector `x`.
"""
basemeasure


"""
    gradlognorm(pdf::ExpFamilyDistribution)

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
    lognorm(pdf::ExpFamilyDistribution)

Return the log-normalization constant of `pdf`
"""
lognorm


"""
    naturalparam(pdf::ExpFamilyDistribution)

Return the natural (a.k.a. the canonical) parameters as a vector.
"""
naturalparam


"""
    stats(pdf::ExpFamilyDistribution, X::AbstractMatrix)

Extract the sufficient statistics of `X` corresponding to type of
`pdf`.
"""
stats


"""
    stdparam(pdf::ExpFamilyDistribution)

Return the standard parameters of `pdf`. If there are multiple
parameters (i.e. a mean and a covariance matrix) the returned value
will be a `namedtuple`.
"""
stdparam


"""
    update!(pdf::ExpFamilyDistribution, naturalparam::AbstractVector)

Update the parameters given a new natural parameter vector.
"""
update!

#######################################################################
# Concrete distributions

export Gamma
export Normal
export PolyaGamma

include("Dirichlet.jl")
include("Gamma.jl")
include("Normal.jl")
include("PolyaGamma.jl")

end

