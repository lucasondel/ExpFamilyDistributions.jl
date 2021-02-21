
#######################################################################
# Parameters interface.

"""
    struct Parameter{T}
        ξ::AbstractVector{T}
        ξ_to_η::Function
        η_to_ξ::Function
    end

Object containing the parameter of a distribution. `T` is
the numerical type of how are stored the parameters (`Float32`,
`Float64`, ...). `ξ` is a vector storing the parameters. `ξ_to_η` and
`η_to_ξ` are functions to convert the stored parameters to their
natural form and vice versa.
"""
struct Parameter{T}
    ξ::AbstractVector{T}
    ξ_to_η::Function
    η_to_ξ::Function
end

"""
    naturalform(param[, ξ])

Returns the natural form of the parameters stored in `ξ`. If `ξ` is not
provided, the function will use `realform(param)` instead.

See also: [`stdform`](@ref), [`realform`](@ref).
"""
naturalform(param, ξ = param.ξ) = param.ξ_to_η(ξ)

"""
    realform(param[, η])

Returns the vector of parameters as stored in `param`. If the natural
parameters `η` is provided, returns their real form.

See also: [`stdform`](@ref), [`naturalform`](@ref).
"""
realform(param, η = naturalform(param)) = param.η_to_ξ(η)

#######################################################################
# Distribution interface

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
    println(io, typeof(dist), ":")

    params = stdparam(dist)
    for (prop, val) in zip(keys(params), params)
        println(io, "  $prop = $val")
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
gradlognorm(p) = FD.gradient(η -> lognorm(p, η), naturalform(p.param))

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

