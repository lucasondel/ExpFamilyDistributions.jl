
"""
    mutable struct Gamma{T} <: ExpFamilyDistribution
        α
        β
    end

Gamma distribution.

# Constructors

    Gamma{T}()

where `T` is the encoding type of the parameters.

    Gamma{T}(α, β)

where `T` is a encoding type of the parameters and `α` and `β` are the
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

#######################################################################
# ExpFamilyDistribution interface

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

function update!(g::Gamma, η)
   g.β = -η[1]
   g.α = η[2]
   return g
end
