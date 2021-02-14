
#######################################################################
# Normal distribution with full covariance matrix

"""
    mutable struct Normal{T,D} <: ExpFamilyDistribution
        μ
        Σ
    end

Normal distribution with full covariance matrix.

# Constructors

    Normal{T,D}()
    Normal(μ[, Σ])

where `T` is the encoding type of the parameters, `D` is the
dimension of the support, `μ` is a vector and `Σ` is a
[**symmetric**](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Symmetric)
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
mutable struct Normal{T,D} <: ExpFamilyDistribution
    μ::Vector{T}
    Σ::Symmetric{T}

    function Normal(μ::AbstractVector{T}, Σ::Symmetric{T}) where T
        if size(μ) ≠ size(Σ)[1] ≠ size(Σ)[2]
            error("Dimension mismatch: size(μ) = $(size(μ)) size(Σ) = $(size(Σ))")
        end
        new{T, length(μ)}(μ, Σ)
    end
end

function Normal{T, D}() where {T, D}
    Normal(zeros(T, D), Symmetric(Matrix{T}(I, D, D)))
end

function Normal(μ::AbstractVector)
    T = eltype(μ)
    D = length(μ)
    Normal(μ, Symmetric(Matrix{T}(I, D, D)))
end

function Base.show(io::IO, ::MIME"text/plain", n::Normal)
    println(io, typeof(n), ":")
    println(io, "  μ = ", n.μ)
    print(io, "  Σ = ", n.Σ)
end

# Split a vector of natural parameters into two components: Λμ and Λ.
_splitnatparams(η, D) = η[1:D], reshape(η[D+1:end], (D, D))

function basemeasure(::Normal{T,D}, x::AbstractVector{T}) where {T,D}
    length(x) == D || throw(DimensionMismatch("expected input dimension $D got $(length(x))"))
    -T(.5) * length(x) * log(T(2π))
end

function gradlognorm(n::Normal; vectorize = true)
    x, xxᵀ = n.μ, n.Σ + n.μ*n.μ'
    vectorize ? vcat(x, vec(xxᵀ)) : (x, xxᵀ)
end
lognorm(n::Normal) = .5 * (logdet(n.Σ) + dot(n.μ, inv(n.Σ), n.μ))
mean(pdf::Normal) = pdf.μ

function naturalparam(n::Normal)
    T = eltype(n.μ)
    Λ = inv(n.Σ)
    vcat(Λ * n.μ, -T(.5) .* vec(Λ))
end

function sample(n::Normal{T,D}, size = 1) where {T,D}
    L = cholesky(n.Σ).L
    [n.μ + L*randn(T, D) for i in 1:size]
end

function stats(::Normal{T,D}, x::AbstractVector{T}) where {T,D}
    length(x) == D || throw(DimensionMismatch("expected input dimension $D got $(length(x))"))
    vcat(x, vec(x*x'))
end

function stdparam(::Normal{T,D}, η::AbstractVector{T}) where {T,D}
    Λμ, nhΛ = _splitnatparams(η, D)
    Λ = Symmetric(-2 * nhΛ)
    Σ = inv(Λ)
    μ = Σ * Λμ
    μ, Σ
end

function update!(n::Normal{T,D}, η::AbstractVector{T}) where {T,D}
    n.μ, n.Σ = stdparam(n, η)
    n
end

#######################################################################
# Normal distribution with diagonal covariance matrix

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
    NormalDiag(μ[, v])

where `T` is the encoding type of the parameters, `D` is the
dimension of the support, `μ` is a vector and `v` is the diagonal of
the covariance matrix.

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
mutable struct NormalDiag{T,D} <: ExpFamilyDistribution
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

function basemeasure(::NormalDiag{T,D}, x::AbstractVector{T}) where {T,D}
    length(x) == D || throw(DimensionMismatch("expected input dimension $D got $(length(x))"))
    -T(.5) * length(x) * log(T(2π))
end

function gradlognorm(n::NormalDiag; vectorize = true)
    x, x² = n.μ, n.v + n.μ.^2
    vectorize ? vcat(x, x²) : (x, x²)
end

function lognorm(n::NormalDiag{T,D}) where {T,D}
    T(.5) * sum(log.(n.v)) + T(.5) * dot(n.μ, (1 ./ n.v) .* n.μ)
end
mean(n::NormalDiag) = n.μ
naturalparam(n::NormalDiag{T,D}) where {T,D} = vcat((T(1) ./ n.v) .* n.μ, -T(.5) .* (1 ./ n.v))

function sample(n::NormalDiag{T,D}, size = 1) where {T,D}
    σ = sqrt.(n.v)
    [n.μ + σ .* randn(T, D) for i in 1:size]
end

stats(::NormalDiag, x::AbstractVector) = vcat(x, x.^2)

function stdparam(n::NormalDiag{T,D}, η::AbstractVector{T}) where {T,D}
    Λμ, nhλ = η[1:D], η[D+1:end]
    v = 1 ./ (-2 * nhλ)
    μ = v .* Λμ
    μ, v
end

function update!(n::NormalDiag, η::AbstractVector)
    n.μ, n.v = stdparam(n, η)
    return n
end

#######################################################################
# δ-Normal distribution

"""
    mutable struct δNormal{T,D} <: δDistribution
        μ
    end

The δ-equivalent of the [`Normal`](@ref) distribution.

# Constructors

    δNormal{T,D}()
    δNormal(μ)

where `T` is the encoding type of the parameters and `D` is the
dimension of the support and `μ` is the location of the Dirac δ pulse.

# Examples
```jldoctest
julia> δNormal{Float32,2}()
δNormal{Float32,2}:
  μ = Float32[0.0, 0.0]

julia> δNormal([1.0, 1.0])
δNormal{Float64,2}:
  μ = [1.0, 1.0]
```
"""
mutable struct δNormal{T,D} <: δDistribution
    μ::Vector{T}

    function δNormal(μ::Vector{T}) where T
        new{T, length(μ)}(μ)
    end
end

δNormal{T,D}() where {T,D} = δNormal(zeros(T,D))

function gradlognorm(n::δNormal; vectorize = true)
    μ, μμᵀ = n.μ, n.μ * n.μ'
    vectorize ? vcat(μ, vec(μμᵀ)) : (μ, μμᵀ)
end

function stdparam(::δNormal{T,D}, η::AbstractVector{T}) where {T,D}
    Λμ, nhΛ = _splitnatparams(η, D)
    Λ = Symmetric(-2 * nhΛ)
    Σ = inv(Λ)
    μ = Σ * Λμ
    μ
end

function update!(n::δNormal, η::AbstractVector)
    n.μ = stdparam(n, η)
    n
end

#######################################################################
# δ-Normal distribution with diagonal covariance matrix

"""
    mutable struct δNormalDiag{T,D} <: δDistribution
        μ
    end

The δ-equivalent of the [`NormalDiag`](@ref) distribution.

# Constructors

    δNormalDiag{T,D}()
    δNormalDiag(μ)

where `T` is the encoding type of the parameters, `D` is the
dimension of the support and `μ` is the location of the Dirac δ pulse.

# Examples
```jldoctest
julia> δNormalDiag{Float32, 2}()
δNormalDiag{Float32,2}:
  μ = Float32[0.0, 0.0]

julia> δNormalDiag([1.0, 1.0])
δNormalDiag{Float64,2}:
  μ = [1.0, 1.0]
```
"""
mutable struct δNormalDiag{T,D} <: δDistribution
    μ::Vector{T}

    δNormalDiag(μ::Vector{T}) where T =  new{T, length(μ)}(μ)
end

δNormalDiag{T, D}() where {T,D} = δNormalDiag(zeros(T, D))

function gradlognorm(n::δNormalDiag; vectorize = true)
    μ, μ² = n.μ, n.μ.^2
    vectorize ? vcat(μ, μ²) : (μ, μ²)
end

function stdparam(n::δNormalDiag{T,D}, η::AbstractVector{T}) where {T,D}
    Λμ, nhλ = η[1:D], η[D+1:end]
    v = 1 ./ (-2 * nhλ)
    μ = v .* Λμ
    μ
end

function update!(n::δNormalDiag, η::AbstractVector)
    n.μ = stdparam(n, η)
    n
end

