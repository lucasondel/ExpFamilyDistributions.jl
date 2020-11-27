
abstract type AbstractNormal <: ExpFamilyDistribution end

# Subtypes should implement:
#   getproperty(n::AbstractNormal, :μ)
#   getproperty(n::AbstractNormal, :Σ)

# Split a vector of natural parameters into two components: Λμ and Λ.
_splitnatparams(η, D) = η[1:D], reshape(η[D+1:end], (D, D))

function Base.show(io::IO, n::AbstractNormal)
    cindent = get(io, :indent, 0)
    println(io, " "^cindent, typeof(n))
    println(io, " "^cindent, "  μ = ", n.μ)
    print(io, " "^cindent, "  Σ = ", n.Σ)
end

#######################################################################
# ExpFamilyDistribution interface

function basemeasure(::AbstractNormal, X::Matrix{T}) where T <: AbstractFloat
    retval = ones(T, size(X, 2))
    retval[:] .= -.5 * size(X, 1) * log(2 * pi)
    retval
end

gradlognorm(n::AbstractNormal) = vcat(n.μ, vec(n.Σ + n.μ * n.μ'))
lognorm(n::AbstractNormal) = .5 * (logdet(n.Σ) + dot(n.μ, inv(n.Σ), n.μ))
mean(pdf::AbstractNormal) = pdf.μ

function naturalparam(n::AbstractNormal)
    Λ = inv(n.Σ)
    vcat(Λ * n.μ, -.5 .* vec(Λ)) end

function stats(::AbstractNormal, X::Matrix{<:AbstractFloat})
    dim1, dim2 = size(X)
    XX = reshape(X, dim1, 1, dim2) .* reshape(X, 1, dim1, dim2)
    vec_XX = reshape(XX, :, dim2)
    vcat(X, vec_XX)
end

function update!(n::AbstractNormal, η::Vector{T}) where T <: AbstractFloat
    D = length(n.μ)
    Λμ, nhΛ = _splitnatparams(η, D)
    Λ = -2 * nhΛ
    n.Σ[:, :] = inv(Symmetric(Λ))
    n.μ[:] = n.Σ * Λμ
    return n
end

#######################################################################
# Concrete implementation Normal with full covariance matrix

mutable struct Normal{T, D} <: AbstractNormal where T <: AbstractFloat
    μ::Vector{T}
    Σ::Matrix{T}

    function Normal(μ::Vector{T}, Σ::Matrix{T}) where T <: AbstractFloat
        if size(μ) ≠ size(Σ)[1] ≠ size(Σ)[2]
            error("Dimension mismatch: size(μ) = $(size(μ)) size(Σ) = $(size(Σ))")
        end
        new{T, length(μ)}(μ, Symmetric(Σ))
    end
end

function Normal(μ::Vector{T}) where T <: AbstractFloat
    D = length(μ)
    Normal(μ, Matrix{T}(I, D, D))
end

function Normal{T, D}() where {T <: AbstractFloat, D}
    Normal(zeros(T, D), Matrix{T}(I, D, D))
end

#######################################################################
# Concrete implementation Normal with diagonal covariance matrix

mutable struct NormalDiag{T, D} <: AbstractNormal where T <: AbstractFloat
    μ::Vector{T}
    v::Vector{T} # Diagonal of the covariance matrix

    function NormalDiag(μ::Vector{T}, v::Vector{T}) where T <: AbstractFloat
        if size(μ) ≠ size(v)
            error("Dimension mismatch: size(μ) = $(size(μ)) size(v) = $(size(v))")
        end
        new{T, length(μ)}(μ, v)
    end
end

# We redefine the show function to avoid allocating the full matrix
# in jupyter notebooks.
function Base.show(io::IO, n::NormalDiag)
    print(io, "$(typeof(n))\n")
    print(io, "  μ = $(n.μ)\n")
    print(io, "  v = $(n.v)")
end

function Base.getproperty(n::NormalDiag, sym::Symbol)
    sym == :Σ ? diagm(n.v) : getfield(n, sym)
end

NormalDiag(μ::Vector{T}) where T<:AbstractFloat = NormalDiag(μ, ones(T, length(μ)))
NormalDiag{T, D}() where {T <: AbstractFloat, D} = NormalDiag(zeros(T, D), ones(T, D))

gradlognorm(n::NormalDiag) = vcat(n.μ, n.v + n.μ.^2)
lognorm(n::NormalDiag) = .5 * sum(log.(n.v)) + sum(n.v .* n.μ.^2)
mean(n::NormalDiag) = n.μ
naturalparam(n::NormalDiag) = vcat((1 ./ n.v) .* n.μ, -.5 .* n.v)
stats(::NormalDiag, X::Matrix{<:AbstractFloat}) = vcat(X, X.^2)

function update!(n::NormalDiag, η::Vector{T}) where T <: AbstractFloat
    D = length(n.μ)
    Λμ, nhλ = η[1:D], η[D+1:end]
    n.v = 1 ./ (-2 * nhλ)
    n.μ = n.v .* Λμ
    return n
end

