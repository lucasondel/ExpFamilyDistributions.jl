abstract type AbstractNormal{T,D} <: ExpFamilyDistribution end

# Subtypes should implement:
#   getproperty(n::AbstractNormal, :μ)
#   getproperty(n::AbstractNormal, :Σ)

# Split a vector of natural parameters into two components: Λμ and Λ.
_splitnatparams(η, D) = η[1:D], reshape(η[D+1:end], (D, D))

function Base.show(io::IO, ::MIME"text/plain", n::AbstractNormal)
    println(io, typeof(n), ":")
    println(io, "  μ = ", n.μ)
    print(io, "  Σ = ", n.Σ)
end

#######################################################################
# ExpFamilyDistribution interface

function basemeasure(::AbstractNormal{T,D}, x::AbstractVector{T}) where {T,D}
    length(x) == D || throw(DimensionMismatch("expected input dimension $D got $(length(x))"))
    -T(.5) * length(x) * log(T(2π))
end

function gradlognorm(n::AbstractNormal; vectorize = true)
    if vectorize
        return vcat(n.μ, vec(n.Σ + n.μ * n.μ'))
    end
    n.μ, n.Σ + n.μ * n.μ'
end
lognorm(n::AbstractNormal) = .5 * (logdet(n.Σ) + dot(n.μ, inv(n.Σ), n.μ))
mean(pdf::AbstractNormal) = pdf.μ

function naturalparam(n::AbstractNormal)
    T = eltype(n.μ)
    Λ = inv(n.Σ)
    vcat(Λ * n.μ, -T(.5) .* vec(Λ))
end

function stats(::AbstractNormal{T,D}, x::AbstractVector{T}) where {T,D}
    length(x) == D || throw(DimensionMismatch("expected input dimension $D got $(length(x))"))
    vcat(x, vec(x*x'))
end

function update!(n::AbstractNormal, η::AbstractVector{T}) where T
    D = length(n.μ)
    Λμ, nhΛ = _splitnatparams(η, D)
    Λ = -2 * nhΛ
    n.Σ = inv(Λ)
    n.μ = n.Σ * Λμ
    n
end

#######################################################################
# Concrete implementation Normal with full covariance matrix

"""
    mutable struct Normal{T,D} <: ExpFamilyDistribution
        μ
        Σ
    end

Normal distribution with full covariance matrix.

# Constructors

    Normal{T,D}()

where `T` is the encoding type of the parameters and `D` is the
dimension of the support.

    Normal(μ[, Σ])

where `μ` is a vector and `Σ` is an [**symmetric**](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Symmetric)
    matrix.

# Examples
```jldoctest
julia> Normal{Float32,2}()
Normal{Float32,2}:
  μ = Float32[0.0, 0.0]
  Σ = Float32[1.0 0.0; 0.0 1.0]

julia> Normal([1.0, 1.0])
Normal{Float64,2}:
  μ = [1.0, 1.0]
  Σ = [1.0 0.0; 0.0 1.0]

julia> using LinearAlgebra; Normal([1.0, 1.0], Symmetric([2.0 0.5; 0.5 1.0]))
Normal{Float64,2}:
  μ = [1.0, 1.0]
  Σ = [2.0 0.5; 0.5 1.0]
```
"""
mutable struct Normal{T,D} <: AbstractNormal{T,D}
    μ::Vector{T}
    Σ::Matrix{T}

    function Normal(μ::AbstractVector{T}, Σ::Symmetric{T}) where T
        if size(μ) ≠ size(Σ)[1] ≠ size(Σ)[2]
            error("Dimension mismatch: size(μ) = $(size(μ)) size(Σ) = $(size(Σ))")
        end
        new{T, length(μ)}(μ, Σ)
    end
end

function Normal(μ::AbstractVector)
    T = eltype(μ)
    D = length(μ)
    Normal(μ, Symmetric(Matrix{T}(I, D, D)))
end

function Normal{T, D}() where {T, D}
    Normal(zeros(T, D), Symmetric(Matrix{T}(I, D, D)))
end

#######################################################################
# Concrete implementation Normal with diagonal covariance matrix

"""
    mutable struct NormalDiag{T,D} <: ExpFamilyDistribution
        μ
        v
    end
Normal distribution with a diagonal covariance matrix. `v` is the
diagonal of the covariance matrix. Note that you can still
access the full covariance matrix by using the property `Σ`.

# Constructors

    NormalDiag{T,D}()

where `T` is the encoding type of the parameters and `D` is the
dimension of the support.

    NormalDiag(μ[, v])

where `μ` is a vector and `v` is the diagonal of the covariance matrix.

# Examples
```jldoctest
julia> NormalDiag{Float32, 2}()
NormalDiag{Float32,2}
  μ = Float32[0.0, 0.0]
  v = Float32[1.0, 1.0]

julia> NormalDiag([1.0, 1.0])
NormalDiag{Float64,2}
  μ = [1.0, 1.0]
  v = [1.0, 1.0]

julia> NormalDiag([1.0, 1.0], [2.0, 1.0])
NormalDiag{Float64,2}
  μ = [1.0, 1.0]
  v = [2.0, 1.0]
```
"""
mutable struct NormalDiag{T,D} <: AbstractNormal{T,D}
    μ::Vector{T}
    v::Vector{T} # Diagonal of the covariance matrix

    function NormalDiag(μ::Vector{T}, v::Vector{T}) where T
        if size(μ) ≠ size(v)
            error("Dimension mismatch: size(μ) = $(size(μ)) size(v) = $(size(v))")
        end
        new{T, length(μ)}(μ, v)
    end
end

function Base.getproperty(n::NormalDiag, sym::Symbol)
    sym == :Σ ? diagm(n.v) : getfield(n, sym)
end

# We redefine the show function to avoid allocating the full matrix
# in jupyter notebooks.
function Base.show(io::IO, ::MIME"text/plain", n::NormalDiag)
    print(io, "$(typeof(n))\n")
    print(io, "  μ = $(n.μ)\n")
    print(io, "  v = $(n.v)")
end

NormalDiag(μ::AbstractVector) = NormalDiag(μ, ones(eltype(μ), length(μ)))
NormalDiag{T, D}() where {T,D} = NormalDiag(zeros(T, D), ones(T, D))

function gradlognorm(n::NormalDiag; vectorize = true)
    if vectorize
        return vcat(n.μ, n.v + n.μ.^2)
    end
    n.μ, n.v + n.μ.^2
end
function lognorm(n::NormalDiag{T,D}) where {T,D}
    T(.5) * sum(log.(n.v)) + T(.5) * dot(n.μ, (1 ./ n.v) .* n.μ)
end
mean(n::NormalDiag) = n.μ
naturalparam(n::NormalDiag{T,D}) where {T,D} = vcat((T(1) ./ n.v) .* n.μ, -T(.5) .* (1 ./ n.v))
stats(::NormalDiag, x::AbstractVector) = vcat(x, x.^2)

function update!(n::NormalDiag, η::AbstractVector)
    D = length(n.μ)
    Λμ, nhλ = η[1:D], η[D+1:end]
    n.v = 1 ./ (-2 * nhλ)
    n.μ = n.v .* Λμ
    return n
end

