
#######################################################################
# Wishart distribution

"""
    mutable struct Wishart{T, D} <: ExpFamilyDistribution
        W
        v
    end

Wishart distribution.

# Constructors

    Wishart{T,D}()
    Wishart(W[, v])

where `T` is the encoding type of the parameters and `W` is a
[**symmetric**](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Symmetric)
DxD matrix.

# Examples
```jldoctest
julia> Wishart{Float32,2}()
Wishart{Float32,2}:
  W = Float32[1.0 0.0; 0.0 1.0]
  v = 2.0

julia> using LinearAlgebra; Wishart(Symmetric([1 0.5; 0.5 1]))
Wishart{Float64,2}:
  W = [1.0 0.5; 0.5 1.0]
  v = 2.0
```
"""
mutable struct Wishart{T,D} <: ExpFamilyDistribution
    W::Symmetric{T}
    v::T

    function Wishart(W::Symmetric{T}, v::T) where T<:Real
        new{T,size(W, 1)}(W, v)
    end
end

function Wishart(W::Symmetric, v::Real)
    T = eltype(W)
    Wishart(W, T(v))
end
Wishart(W::Symmetric) = Wishart(W, size(W,1))

function Wishart{T,D}() where {T<:Real,D}
    W = Symmetric(Matrix{T}(I,D,D))
    v = T(D)
    Wishart(W, v)
end

function Base.show(io::IO, ::MIME"text/plain", w::Wishart)
    println(io, typeof(w), ":")
    println(io, "  W = ", w.W)
    print(io, "  v = ", w.v)
end

function basemeasure(w::Wishart, X::Symmetric)
    D = size(X, 1)
    -.5*(D-1)*logdet(X) - .25*D*(D-1)log(pi)
end

function gradlognorm(w::Wishart{T,D}; vectorize = true) where {T,D}
    ∂η₁ = w.v*w.W
    ∂η₂ = sum([digamma((T(w.v+1-i)/2)) for i in 1:D]) + T(D*log(2)) + logdet(w.W)
    if vectorize
        return vcat(vec(∂η₁), ∂η₂)
    end
    ∂η₁, ∂η₂
end

function lognorm(w::Wishart{T,D}) where {T,D}
    .5*( w.v*logdet(w.W) +  w.v*D*log(2) ) + sum([loggamma((w.v+1-i)/2) for i in 1:D])
end

function naturalparam(w::Wishart{T,D}) where {T,D}
    η₁ = -T(.5) * inv(w.W)
    η₂ = T(.5) * w.v
    vcat(vec(η₁), η₂)
end

mean(w::Wishart) = w.v * w.W
stats(w::Wishart{T}, X::Symmetric{T}) where T = vcat(vec(X), logdet(X))

function stdparam(::Wishart{T,D}, η::AbstractVector{T}) where {T,D}
    W = inv(-2*Symmetric(reshape(η[1:end-1],D,D)))
    v = 2*η[end]
    W, v
end

function update!(w::Wishart, η)
    w.W, w.v = stdparam(w, η)
   return w
end

#######################################################################
# δ-Wishart distribution

"""
    mutable struct δWishart{T,D} <: δDistribution
        μ
    end

The δ-equivaltent of the [`Wishart`](@ref) distribution.

# Constructors

    δWishart{T,D}()
    δWishart(μ)

where `T` is the encoding type of the parameters and `μ` is the
location of the Dirac δ pulse.

# Examples
```jldoctest
julia> δWishart{Float32,2}()
δWishart{Float32,2}:
  μ = Float32[1.0 0.0; 0.0 1.0]

julia> using LinearAlgebra; δWishart(Symmetric([1 0.5; 0.5 1]))
δWishart{Float64,2}:
  μ = [1.0 0.5; 0.5 1.0]
```
"""
mutable struct δWishart{T,D} <: δDistribution
    μ::Symmetric{T}

    function δWishart(μ::Symmetric{T}) where T<:Real
        new{T,size(μ,1)}(μ)
    end
end

δWishart{T,D}() where {T<:Real,D} = δWishart(Symmetric(Matrix{T}(I,D,D)))

function gradlognorm(w::δWishart; vectorize = true)
    vectorize ? vcat(vec(w.μ), logdet(w.μ)) : (w.μ, logdet(w.μ))
end

function stdparam(::δWishart{T,D}, η) where {T,D}
    W = inv(-2*Symmetric(reshape(η[1:end-1],D,D)))
    v = 2*η[end]
    v*W
end

function update!(w::δWishart, η)
    μ = stdparam(w, η)
    w.μ = μ
    w
end

