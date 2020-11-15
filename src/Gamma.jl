
abstract type AbstractGamma <: ExpFamilyDistribution end

# Subtypes should implement:
#   getproperty(n::AbstractNormal, :α)
#   getproperty(n::AbstractNormal, :β)

function Base.show(io::IO, g::AbstractGamma)
    print(io, "$(typeof(g))\n")
    print(io, "  α = $(g.α)\n")
    print(io, "  β = $(g.β)")
end

#######################################################################
# ExpFamilyDistribution interface

basemeasure(::AbstractGamma, x::Vector{<:AbstractFloat}) = -log.(x)
gradlognorm(g::AbstractGamma) = vcat(g.α / g.β, digamma(g.α) - log(g.β))
lognorm(g::AbstractGamma) = sum(loggamma.(g.α) .- g.α .* log.(g.β))
naturalparam(g::AbstractGamma) = vcat(-g.β, g.α)
mean(g::AbstractGamma) = g.α / g.β
stats(::AbstractGamma, x::Vector{<:AbstractFloat}) = vcat(x', log.(x)')

function update!(g::AbstractGamma, η::Vector{<:AbstractFloat})
   g.β = -η[1]
   g.α = η[2]
   return g
end

#######################################################################
# Concrete implementation

"""
    Gamma <: AbstractGamma
    Gamma([Float32 | Float64,] α, β)

Gamma density.

# Example:
julia> Gamma
"""
mutable struct Gamma{T} <: AbstractGamma where T <: AbstractFloat
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
