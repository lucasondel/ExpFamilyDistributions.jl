
using SpecialFunctions: loggamma, digamma

"""
    Gamma <: ExpFamilyDistribution
    Gamma([Float32 | Float64,] α, β)

Gamma density.

# Example:
julia> Gamma
"""
mutable struct Gamma{T} <: ExpFamilyDistribution where T <: AbstractFloat
    α::T
    β::T

    function Gamma(α::T, β::T) where T <: AbstractFloat
       new{T}(α, β)
   end
end

Gamma(T::Type{<:AbstractFloat}, α::Real, β::Real) = Gamma(T(α), T(β))
Gamma(α::Real, β::Real) = Gamma(Float64, α, β)
Gamma(α::Real) = Gamma(α, 1)
Gamma() = Gamma(1, 1)

function Base.show(io::IO, g::Gamma)
    print(io, "$(typeof(g))\n")
    print(io, "  α = $(g.α)\n")
    print(io, "  β = $(g.β)")
end


#######################################################################
# ExpFamilyDistribution interface


basemeasure(::Gamma, x::Vector{<:AbstractFloat}) = -log.(x)

function gradlognorm(g::Gamma)
    α, β = stdparam(g)
    vcat(α / β, digamma(α) - log(β))
end

function lognorm(g::Gamma)
   α, β = stdparam(g)
   sum(loggamma.(α) .- α .* log.(β))
end

naturalparam(g::Gamma) = vcat(-g.β, g.α)
mean(g::Gamma) = g.α / g.β
stats(::Gamma, x::Vector{<:AbstractFloat}) = vcat(x', log.(x)')
stdparam(g::Gamma) = (α = g.α, β = g.β)

function update!(g::Gamma{T}, η::Vector{T}) where {T <: AbstractFloat}
   g.β = -η[1]
   g.α = η[2]
   return g
end

