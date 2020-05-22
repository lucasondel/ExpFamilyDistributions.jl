
struct Normal{T, D} <: ExpFamilyDistribution where T <: AbstractFloat
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

function Base.show(io::IO, n::Normal)
    print(io, "$(typeof(n))\n")
    print(io, "  μ = $(n.μ)\n")
    print(io, "  Σ = $(n.Σ)")
end


# Split a vector of natural parameters into two components: Λμ and Λ.
_splitnatparams(η, D) = η[1:D], reshape(η[D+1:end], (D, D))


#######################################################################
# ExpFamilyDistribution interface


function basemeasure(::Normal, X::Matrix{T}) where T <: AbstractFloat
    retval = ones(T, size(X, 2))
    retval[:] .= -.5 * size(X, 1) * log(2 * pi)
    retval
end

function gradlognorm(pdf::Normal)
    μ, Σ = stdparam(pdf)
    vcat(μ, vec(Σ + μ * μ'))
end

lognorm(pdf::Normal) = .5 * (logdet(pdf.Σ) + dot(pdf.μ, inv(pdf.Σ), pdf.μ))

mean(pdf::Normal) = pdf.μ

function naturalparam(pdf::Normal)
    Λ = inv(pdf.Σ)
    vcat(Λ * pdf.μ, -.5 .* vec(Λ))
end

function stats(::Normal, X::Matrix{<:AbstractFloat})
    dim1, dim2 = size(X)
    XX = reshape(X, dim1, 1, dim2) .* reshape(X, 1, dim1, dim2)
    vec_XX = reshape(XX, :, dim2)
    vcat(X, vec_XX)
end

stdparam(pdf::Normal) = (μ=pdf.μ, Σ=pdf.Σ)

function update!(pdf::Normal{T, D}, η::Vector{T}) where {T <: AbstractFloat, D}
    Λμ, nhΛ = _splitnatparams(η, D)
    Λ = -2 * nhΛ
    pdf.Σ[:, :] = inv(Symmetric(Λ))
    pdf.μ[:] = pdf.Σ * Λμ
    return nothing
end

