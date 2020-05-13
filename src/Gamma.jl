
using SpecialFunctions: loggamma, digamma

"""
    Gamma <: ExpFamilyDistribution

Gamma density.
"""
struct Gamma{D} <: ExpFamilyDistribution
   α::AbstractVector
   β::AbstractVector

   function Gamma(α::AbstractVector, β::AbstractVector)
       if size(α) ≠ size(β)
           error("Dimension mismatch: size(α) = $(size(α)) size(β) = $size(\beta))
")
       end
       new{length(α)}(α, β)
   end
end

Gamma{D}(α::Real, β::Real) where D = Gamma(α .* ones(Float64, D), β .* ones(Float64, D))
Gamma{D}(α::Real = 1) where D = Gamma{D}(α, α)


function Base.show(io::IO, g::Gamma)
    print(io, "$(typeof(g))\n")
    print(io, "  α = $(g.α)\n")
    print(io, "  β = $(g.β)")
end


#######################################################################
# ExpFamilyDistribution interface


function basemeasure(::Gamma, X::AbstractMatrix)
    - dropdims(sum(log.(X), dims=1), dims=1)
end


function gradlognorm(g::Gamma)
    α, β = stdparam(g)
    vcat(α ./ β, digamma.(α) - log.(β))
end


function lognorm(g::Gamma)
   α, β = stdparam(g)
   sum(loggamma.(α) .- α .* log.(β))
end


naturalparam(g::Gamma) = vcat(-g.β, g.α)
mean(g::Gamma) = g.α ./ g.β
stats(::Gamma, X::AbstractMatrix) = vcat(X, log.(X))
stdparam(g::Gamma) = (α = g.α, β = g.β)


function update!(g::Gamma{D}, η::AbstractVector) where D
   g.α[:] = η[D+1:end]
   g.β[:] = -η[1:D]
   return nothing
end

