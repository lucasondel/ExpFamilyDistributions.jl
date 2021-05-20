# SPDX-License-Identifier: MIT

"""
    abstract type Distribution end

Supertype for distributions member of the exponential family.
"""
abstract type Distribution end

function Base.getproperty(dist::Distribution, sym::Symbol)
    # Check first the fields to avoid computing the std-params for
    # every property access.
    if sym in fieldnames(typeof(dist))
        return getfield(dist, sym)
    end

    param = stdparam(dist)
    if sym ∈ keys(param)
        return getfield(param, sym)
    end

    # Raise an error.
    getfield(dist, sym)
end

function Base.show(io::IO, ::MIME"text/plain", dist::Distribution)
    print(io, typeof(dist), ":")

    params = stdparam(dist)
    for (prop, val) in zip(keys(params), params)
        println(io)
        print(io, "  $prop = $val")
    end
end

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
gradlognorm(p) = gradient(η -> lognorm(p, η), naturalform(p.param))[1]

"""
    kldiv(q::T, p::T[, μ = gradlognorm(q)]) where T<:Distribution

Compute the KL-divergence between two distributions of the same type
(i.e. `kldiv(Normal, Normal)`, `kldiv(Dirichlet, Dirichlet)`, ...). You
can specify directly the expectation of the sufficient statistics `μ`.
"""
function kldiv(q::T, p::T; μ = gradlognorm(q)) where T<:Distribution
    q_η, p_η = naturalform(q.param), naturalform(p.param)
    lognorm(p) - lognorm(q) - dot(p_η .- q_η, μ)
end

"""
    lognorm(p)

Returns the log-normalization constant of `p`.
"""
lognorm

"""
    sample(p, n=1)

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

