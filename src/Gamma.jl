
using SpecialFunctions: loggamma, digamma

"""
    Gamma <: ExpFamilyDistribution

Gamma density.
"""
struct Gamma{T, D} <: ExpFamilyDistribution where T <: AbstractFloat
    α::Vector{T}
    β::Vector{T}

    function Gamma(α::Vector{T}, β::AbstractVector{T}) where T <: AbstractFloat
       if size(α) ≠ size(β)
           error("Dimension mismatch: size(α) = $(size(α)) size(β) = $size(\beta))
")
       end
       new{T, length(α)}(α, β)
   end
end

Gamma(α::Vector{T}) where T <: AbstractFloat = Gamma(α, ones(T, length(α)))
Gamma{T, D}() where {T <: AbstractFloat, D} = Gamma(ones(T, D), ones(T, D))

function Base.show(io::IO, g::Gamma)
    print(io, "$(typeof(g))\n")
    print(io, "  α = $(g.α)\n")
    print(io, "  β = $(g.β)")
end


#######################################################################
# ExpFamilyDistribution interface


function basemeasure(::Gamma, X::Matrix{<:AbstractFloat})
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
stats(::Gamma, X::Matrix{<:AbstractFloat}) = vcat(X, log.(X))
stdparam(g::Gamma) = (α = g.α, β = g.β)

function update!(g::Gamma{T, D}, η::Vector{T}) where {T <: AbstractFloat, D}
   g.α[:] = η[D+1:end]
   g.β[:] = -η[1:D]
   return nothing
end

