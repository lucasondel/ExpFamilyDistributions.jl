
#######################################################################
# Gamma distribution

"""
    mutable struct Gamma{T} <: ExpFamilyDistribution
        α
        β
    end

Gamma distribution.

# Constructors

    Gamma{T}()
    Gamma{T}(α, β)

where `T` is the encoding type of the parameters. `α` and `β` are the
parameters of the distribution.

# Examples
```jldoctest
julia> Gamma{Float32}()
Gamma{Float32}:
  α = 1.0
  β = 1.0

julia> Gamma{Float64}(1, 2)
Gamma{Float64}:
  α = 1.0
  β = 2.0
```
"""
mutable struct Gamma{T} <: ExpFamilyDistribution
    α::T
    β::T

    function Gamma{T}(α::T, β::T) where T<:Real
       new{T}(α, β)
   end
end

function Base.show(io::IO, g::Gamma)
    println(io, typeof(g), ":")
    println(io, "  α = ", g.α)
    print(io, "  β = ", g.β)
end

Gamma{T}(α::Real, β::Real) where T = Gamma{T}(T(α), T(β))
Gamma{T}() where T<:Real = Gamma{T}(1, 1)

basemeasure(::Gamma, x) = -log(x)
function gradlognorm(g::Gamma; vectorize = true)
    if vectorize
        vcat(g.α / g.β, digamma(g.α) - log(g.β))
    end
    g.α / g.β, digamma(g.α) - log(g.β)
end
lognorm(g::Gamma) = loggamma(g.α) - g.α * log.(g.β)
naturalparam(g::Gamma) = vcat(-g.β, g.α)
mean(g::Gamma) = g.α / g.β
stats(::Gamma{T}, x) where T = T[x, log(x)]

function stdparam(::Gamma{T}, η::AbstractVector{T}) where T
    η[2], -η[1]
end

function update!(g::Gamma, η)
    g.α, g.β = stdparam(g, η)
   return g
end

#######################################################################
# δ-Gamma distribution

"""
    mutable struct δGamma{T} <: δDistribution
        μ
    end

The δ-equivaltent of the [`Gamma`](@ref) distribution.

# Constructors

    δGamma{T}()
    δGamma{T}(μ)

where `T` is the encoding type of the parameters and `μ` is the
location of the Dirac δ pulse.

# Examples
```jldoctest
julia> δGamma{Float32}()
δGamma{Float32}:
  μ = 1.0

julia> δGamma{Float64}(2)
δGamma{Float64}:
  μ = 2.0
```
"""
mutable struct δGamma{T} <: δDistribution
    μ::T

    function δGamma{T}(μ::Real) where T<:Real
        μ ≥ 0 || throw(ArgumentError("Expected μ ≥ 0"))
        new{T}(T(μ))
    end
end

δGamma{T}() where T<:Real = δGamma{T}(1)

function gradlognorm(g::δGamma; vectorize = true)
    vectorize ? vcat(g.μ, log(g.μ)) : (g.μ, log(g.μ))
end

function stdparam(::δGamma{T}, η::AbstractVector{T}) where T
    (η[2]-1) / -η[1]
end

function update!(g::δGamma, η)
    μ = stdparam(g, η)
    μ ≥ 0 || throw(ArgumentError("Expected μ ≥ 0"))
    g.μ = μ
    g
end

