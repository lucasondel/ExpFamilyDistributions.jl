
#######################################################################
# Wishart distribution

"""
    mutable struct Wishart{T, D} <: ExpFamilyDistribution
        diagW
        trilW
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
    diagW::Vector{T}
    trilW::Vector{T}
    v::T

    function Wishart(W::Symmetric{T}, v::T) where T<:Real
        new{T,size(W, 1)}(diag(W), vec_tril(W), v)
    end
end

function Base.getproperty(w::Wishart{T,D}, sym::Symbol) where {T,D}
    if sym == :W
        W = zeros(T, D, D)
        trilW = inv_vec_tril(w.trilW)
        W = Diagonal(w.diagW) + trilW + trilW'
        return W
    end
    getfield(w, sym)
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

function gradlognorm(w::Wishart{T,D}) where {T,D}
    ∂η₁ = w.v*w.diagW
    ∂η₂ = w.v*w.trilW
    ∂η₃ = sum([digamma((T(w.v+1-i)/2)) for i in 1:D]) + T(D*log(2)) + logdet(w.W)
    vcat(∂η₁, ∂η₂, ∂η₃)
end

function lognorm(w::Wishart{T,D}) where {T,D}
    .5*( w.v*logdet(w.W) +  w.v*D*log(2) ) + sum([loggamma((w.v+1-i)/2) for i in 1:D])
end

function naturalparam(w::Wishart{T,D}) where {T,D}
    M = inv(w.W)
    η₁ = -T(.5)*diag(M)
    η₂ = vec_tril(M)
    η₃ = T(.5) * w.v
    vcat(η₁, η₂, η₃)
end

mean(w::Wishart) = w.v * w.W
stats(w::Wishart{T}, X::Symmetric{T}) where T = vcat(diag(X), vec_tril(X), logdet(X))

function sample(w::Wishart, size = 1)
    w_ = Dists.Wishart(w.v, PDMat(w.W))
    [rand(w_) for i in 1:size]
end

function _splitgrad_wishart(D, μ::AbstractVector)
    diag_∂₁ = μ[1:D]
    tril_∂₁ = inv_vec_tril(μ[D+1:end-1])
    ∂₁ = Symmetric(Diagonal(diag_∂₁) + tril_∂₁ + tril_∂₁')
    ∂₁, μ[end]
end
splitgrad(w::Wishart{T,D}, μ::AbstractVector) where {T,D} = _splitgrad_wishart(D, μ)

function stdparam(::Wishart{T,D}, η::AbstractVector{T}) where {T,D}
    diag_invW = -T(2)*η[1:D]
    tril_invW = -inv_vec_tril(η[D+1:end-1])
    invW = Symmetric(Diagonal(diag_invW) + tril_invW + tril_invW')
    W = inv(invW)
    v = 2*η[end]
    W, v
end

function update!(w::Wishart, η)
    W, w.v = stdparam(w, η)
    w.diagW = diag(W)
    w.trilW = vec_tril(W)
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

gradlognorm(w::δWishart) = vcat(diag(w.μ), vec_tril(w.μ), logdet(w.μ))

splitgrad(w::δWishart{T,D}, μ::AbstractVector) where {T,D} = _splitgrad_wishart(D, μ)

function stdparam(::δWishart{T,D}, η) where {T,D}
    diag_invW = η[1:D]
    tril_invW = inv_vec_tril(η[D+1:end-1])
    invW = Symmetric(Diagonal(diag_invW) + tril_invW + tril_invW')
    W = inv(-2*invW)
    v = 2*η[end]
    v*W
end

function update!(w::δWishart, η)
    μ = stdparam(w, η)
    w.μ = μ
    w
end

