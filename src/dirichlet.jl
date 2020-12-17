
"""
    mutable struct Dirichlet{T,D} <: ExpFamilyDistribution
        α
    end

Dirichlet distribution.

# Constructors

    Dirichlet{T,D}()

where `T` is the encoding type of the parameters and `D` is the
dimension of the support.

    Dirichlet(α)

where `T` is a encoding type of the parameters and `α` is a vector of
parameters

# Examples
```jldoctest
julia> Dirichlet{Float32, 2}()
Dirichlet{Float32,2}:
  α = Float32[1.0, 1.0]

julia> Dirichlet([1.0, 2.0, 3.0])
Dirichlet{Float64,3}:
  α = [1.0, 2.0, 3.0]
```
"""
mutable struct Dirichlet{T,D} <: ExpFamilyDistribution
    α::Vector{T}

    function Dirichlet(α::Vector{T}) where T
        new{T, length(α)}(α)
    end
end

function Base.show(io::IO, d::Dirichlet)
    println(io, typeof(d), ":")
    print(io, "  α = ", d.α)
end

Dirichlet{T,D}() where {T, D} = Dirichlet(ones(T, D))

#######################################################################
# ExpFamilyDistribution interface

basemeasure(::Dirichlet, x::AbstractVector) = -log.(x)
gradlognorm(d::Dirichlet) = digamma.(d.α) .- digamma(sum(d.α))
stats(::Dirichlet, x::AbstractVector) = log.(x)
lognorm(d::Dirichlet) = sum(loggamma.(d.α)) - loggamma(sum(d.α))
mean(d::Dirichlet) = d.α ./ sum(d.α)
naturalparam(d::Dirichlet) = d.α
function update!(d::Dirichlet, η::AbstractVector)
    d.α = η
    d
end

