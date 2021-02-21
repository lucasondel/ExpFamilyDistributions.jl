
#######################################################################
# Parameter of the Gamma distribution.

function DefaultGammaParameter(T, α, β)
    Parameter{T}(vcat(-β, α), identity, identity)
end

#######################################################################
# Gamma distribution

"""
    struct Gamma <: Distribution
        param::Parameter{T} where T
    end

Gamma distribution.

# Constructors

    Gamma(T=Float64)
    Gamma(T=Float64, α, β)

where `T` is the encoding type of the parameters. `α` and `β` are the
parameters of the distribution.

# Examples
```jldoctest
julia> Gamma(Float32)
Gamma:
  α = 1.0
  β = 1.0

julia> Gamma(1, 2)
Gamma:
  α = 1.0
  β = 2.0
```
"""
struct Gamma <:  Distribution
    param::Parameter{T} where T
end

Gamma(T, α, β) = Gamma(DefaultGammaParameter(T, α, β))
Gamma(α::Real, β::Real) = Gamma(Float64, α, β)
Gamma(T::Type = Float64) = Gamma(T, 1, 1)

#######################################################################
# Distribution interface

basemeasure(::Gamma, x) = -log(x)

function lognorm(g::Gamma, η::AbstractVector = naturalform(g.param))
    η₁, η₂ = η
    #loggamma(g.α) - g.α * log.(g.β)
    loggamma(η₂) - η₂ * log(-η₁)
end

function sample(g::Gamma, size)
    g_ = Dists.Gamma(g.α, 1/g.β)
    [Dists.rand(g_) for i in 1:size]
end

splitgrad(g::Gamma, μ::AbstractVector) = μ[1], μ[2]

stats(::Gamma, x) = [x, log(x)]

function stdparam(g::Gamma, η::AbstractVector{T} = naturalform(g.param)) where T
    (α = η[2], β = -η[1])
end

