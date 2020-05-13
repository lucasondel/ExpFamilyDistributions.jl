
struct Normal{D} <: ExpFamilyDistribution
    μ::AbstractVector
    Σ::AbstractMatrix

    function Normal(μ::AbstractVector, Σ::AbstractMatrix)
        if size(μ) ≠ size(Σ)[1] ≠ size(Σ)[2]
            error("Dimension mismatch: size(μ) = $(size(μ)) size(Σ) = $(size(Σ))")
        end
        new{length(μ)}(μ, Σ)
    end
end


function Normal(μ::AbstractVector)
    D = length(μ)
    Normal(μ, Matrix{eltype(μ)}(I, D, D))
end


function Normal{D}() where D
    Normal(zeros(Float64, D), Matrix{Float64}(I, D, D))
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


function basemeasure(::Normal, X::AbstractMatrix)
    -.5 * size(X)[1] * log(2 * pi)
end


function gradlognorm(pdf::Normal)
    μ, Σ = stdparam(pdf)
    vcat(μ, vec(Σ + μ * μ'))
end


function lognorm(pdf::Normal{D}) where D
    η1, η2 = _splitnatparams(naturalparam(pdf), D)
    -0.25 * (dot(η1, inv(η2), η1) - .5 * logdet(-2 * η2))
end

mean(pdf::Normal) = pdf.μ

function naturalparam(pdf::Normal)
    Λ = inv(Symmetric(pdf.Σ))
    vcat(Λ * pdf.μ, -.5 .* vec(Λ))
end

function stats(::Normal, X::AbstractMatrix)
    dim1, dim2 = size(X)
    XX = reshape(X, dim1, 1, dim2) .* reshape(X, 1, dim1, dim2)
    vec_XX = reshape(XX, :, dim2)
    vcat(X, vec_XX)
end

stdparam(pdf::Normal) = (μ=pdf.μ, Σ=pdf.Σ)

function update!(pdf::Normal{D}, η::AbstractVector) where D
    Λμ, nhΛ = _splitnatparams(η, D)
    Λ = -2 * nhΛ
    pdf.Σ[:, :] = inv(Symmetric(Λ))
    pdf.μ[:] = pdf.Σ * Λμ
    return nothing
end

