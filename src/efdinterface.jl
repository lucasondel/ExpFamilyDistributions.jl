
#######################################################################
# ExpFamilyDistribution interface

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
    kldiv(q::T, p::T[, μ = gradlognorm(q)]) where T<:ExpFamilyDistribution

Compute the KL-divergence between two distributions of the same type
(i.e. `kldiv(Normal, Normal)`, `kldiv(Dirichlet, Dirichlet)`, ...). You
can specify directly the expectation of the sufficient statistics `μ`.
"""
function kldiv(q::T, p::T; μ = gradlognorm(q)) where T<:ExpFamilyDistribution
    q_η, p_η = naturalparam(q), naturalparam(p)
    lognorm(p) - lognorm(q) - dot(p_η .- q_η, μ)
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
    sample(p[, n=1])

Draw `n` samples from the distribution `p`.
"""
sample

"""
    splitgrad(p, μ)

Split the gradient of the log-normalizer into its "standard" components.
For instance, for the Normal distribution, the output will be the
expected value of \$x\$ and \$xxᵀ\$.
"""
splitgrad

"""
    stats(p, x)

Returns the sufficient statistics of `x` for the distribution `p`.
"""
stats

"""
    stdparam(p, η)

Returns the standard parameters corresponding to the natural parameters
`η` for distributions with the same type of `p`.
"""
stdparam

"""
    update!(p, η)

Updates the parameters given a new natural parameter `η`.
"""
update!
