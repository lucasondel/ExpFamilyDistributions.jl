
#######################################################################
# Parameter of the Dirichlet distribution.


function DefaultDirichletParameter(α::AbstractVector{T}) where T
    Parameter{T}(α, identity, identity)
end

#######################################################################
# Dirichlet distribution

"""
    struct Dirichlet{D} <: Distribution
        param
    end

Dirichlet distribution.

# Constructors

    Dirichlet{D}(T=Float64)
    Dirichlet(α)

where `T` is the encoding type of the parameters and `D` is the
dimension of the support and `α` is a vector of parameters.

# Examples
```jldoctest
julia> Dirichlet{2}(Float32)
Dirichlet{2}:
  α = Float32[1.0, 1.0]

julia> Dirichlet([1.0, 2.0, 3.0])
Dirichlet{3}:
  α = [1.0, 2.0, 3.0]
```
"""
struct Dirichlet{D} <: Distribution
    param::Parameter{T} where T
end

Dirichlet(α::AbstractVector) = Dirichlet{length(α)}(DefaultDirichletParameter(α))
Dirichlet{D}(T::Type = Float64) where D = Dirichlet(ones(T, D))

#######################################################################
# Distribution interface.

basemeasure(::Dirichlet, x::AbstractVector) = -log.(x)

function lognorm(d::Dirichlet, η::AbstractVector = naturalform(d.param))
    sum(loggamma.(η)) - loggamma(sum(η))
end

function sample(d::Dirichlet, size)
    α = stdparam(d)
    d_ = Dists.Dirichlet(d.α)
    [Dists.rand(d_) for i in 1:size]
end

# vectorization is effect less
splitgrad(d::Dirichlet, μ) = identity(μ)

stats(::Dirichlet, x::AbstractVector) = log.(x)

function stdparam(d::Dirichlet, η::AbstractVector = naturalform(d.param))
    (α = η,)
end

