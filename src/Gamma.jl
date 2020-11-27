
abstract type AbstractGamma <: ExpFamilyDistribution end

# Subtypes should implement:
#   getproperty(n::AbstractNormal, :α)
#   getproperty(n::AbstractNormal, :β)

function Base.show(io::IO, g::AbstractGamma)
    cindent = get(io, :indent, 0)
    println(io, " "^cindent, typeof(g), ":")
    println(io, " "^cindent, "  α = ", g.α)
    print(io, " "^cindent, "  β = ", g.β)
end

#######################################################################
# ExpFamilyDistribution interface

basemeasure(::AbstractGamma, x::Vector{<:AbstractFloat}) = -log.(x)
function gradlognorm(g::AbstractGamma; vectorize = true)
    if vectorize
        vcat(g.α / g.β, digamma(g.α) - log(g.β))
    end
    g.α / g.β, digamma(g.α) - log(g.β)
end
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

    function Gamma{T}(α::T, β::T) where T<:AbstractFloat
       new{T}(α, β)
   end
end

Gamma{T}(α::Real, β::Real) where T<:AbstractFloat = Gamma{T}(T(α), T(β))
Gamma{T}(α::Real) where T<:AbstractFloat = Gamma{T}(α, 1)
Gamma{T}() where T<:AbstractFloat = Gamma{T}(1, 1)

