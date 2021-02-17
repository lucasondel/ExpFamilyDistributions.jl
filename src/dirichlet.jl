
#######################################################################
# Dirichlet distribution

"""
    mutable struct Dirichlet{T,D} <: ExpFamilyDistribution
        α
    end

Dirichlet distribution.

# Constructors

    Dirichlet{T,D}()
    Dirichlet(α)

where `T` is the encoding type of the parameters and `D` is the
dimension of the support and `α` is a vector of parameters.

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

function Base.show(io::IO, ::MIME"text/plain", d::Dirichlet)
    println(io, typeof(d), ":")
    print(io, "  α = ", d.α)
end

Dirichlet{T,D}() where {T, D} = Dirichlet(ones(T, D))

basemeasure(::Dirichlet, x::AbstractVector) = -log.(x)

gradlognorm(d::Dirichlet) = digamma.(d.α) .- digamma(sum(d.α))

lognorm(d::Dirichlet) = sum(loggamma.(d.α)) - loggamma(sum(d.α))
mean(d::Dirichlet) = d.α ./ sum(d.α)
naturalparam(d::Dirichlet) = d.α

function sample(d::Dirichlet, size=1)
    d_ = Dists.Dirichlet(d.α)
    [Dists.rand(d_) for i in 1:size]
end

# vectorization is effect less
splitgrad(d::Dirichlet, μ::AbstractVector) = identity(μ)

stats(::Dirichlet, x::AbstractVector) = log.(x)

function stdparam(::Dirichlet{T}, η::AbstractVector{T}) where T
    η
end

function update!(d::Dirichlet, η::AbstractVector)
    d.α = stdparam(d, η)
    d
end

#######################################################################
# δ-Dirichlet distribution

"""
    mutable struct Dirichlet{T,D} <: δDistribution
        μ
    end

The δ-equivalent of the [`Dirichlet`](@ref) distribution.

# Constructors

    Dirichlet{T,D}()
    Dirichlet(μ)

where `T` is the encoding type of the parameters and `D` is the
dimension of the support and `μ` is the location of the Dirac δ pulse.

# Examples
```jldoctest
julia> δDirichlet{Float32, 2}()
δDirichlet{Float32,2}:
  μ = Float32[0.5, 0.5]

julia> δDirichlet(Float32[1/2, 1/2])
δDirichlet{Float32,2}:
  μ = Float32[0.5, 0.5]
```
"""
mutable struct δDirichlet{T,D} <: δDistribution
    μ::Vector{T}

    function δDirichlet(μ::Vector{T}) where T
        sum(μ) ≈ 1 || throw(ArgumentError("input of the δDirichlet should sum up to one."))
        new{T, length(μ)}(μ)
    end
end

δDirichlet{T,D}() where {T, D} = δDirichlet(ones(T, D) / D)

gradlognorm(d::δDirichlet; vectorize = true) = log.(d.μ)

splitgrad(d::δDirichlet, μ::AbstractVector) = identity(μ)

function stdparam(::δDirichlet{T}, η::AbstractVector{T}) where T
    (η .- 1) / sum(η .- 1)
end

function update!(d::δDirichlet, η::AbstractVector)
    all(η .≥ 1) || throw(ArgumentError("Expected η .≥ 1"))
    d.μ = stdparam(d, η)
end

