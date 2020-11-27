
abstract type AbstractDirichlet <: ExpFamilyDistribution end

# Subtypes should implement:
#   getproperty(n::AbstractDirihlet, :α)

function Base.show(io::IO, d::AbstractDirichlet)
    cindent = get(io, :indent, 0)
    print(io, " "^cindent, typeof(g))
    print(io, " "^cindent, "  α = ", d.α)
end

#######################################################################
# ExpFamilyDistribution interface

basemeasure(::AbstractDirichlet, X::Matrix{T}) where T <: AbstractFloat = zeros(T, size(X, 2))
gradlognorm(d::AbstractDirichlet) = digamma.(d.α) .- digamma(sum(d.α))
stats(::AbstractDirichlet, X::Matrix{T}) where T <: AbstractFloat = log.(X)
lognorm(d::AbstractDirichlet) = sum(loggamma.(d.α)) - loggamma(sum(d.α))
mean(d::AbstractDirichlet) = d.α ./ sum(d.α)
naturalparam(d::AbstractDirichlet) = d.α .- 1
function update!(d::AbstractDirichlet, η::Vector{T}) where T<:AbstractFloat
    d.α = η .+ 1
    d
end

#######################################################################
# Concrete implementation Normal with diagonal covariance matrix

mutable struct Dirichlet{T, D} <: AbstractDirichlet where T <: AbstractFloat
    α::Vector{T}

    function Dirichlet(α::Vector{T}) where T <: AbstractFloat
        new{T, length(α)}(α)
    end
end

Dirichlet{T, D}() where {T <: AbstractFloat, D} = Dirichlet(ones(T, D))

