
using StatsFuns

struct PolyaGamma{T, D} <: ExpFamilyDistribution where T <: AbstractFloat
    b::Vector{T}
    c::Vector{T}

    function PolyaGamma(b::Vector{T}, c::Vector{T}) where T <: AbstractFloat
        if size(b) ≠ size(c)
            error("Dimension mismathc: size(b) = $(size(b)) ≠ size(c) = $(size(c))")
        end
        new{T, length(b)}(b, c)
    end
end

PolyaGamma(b::Vector{T}) where T <: AbstractFloat = PolyaGamma(b, zeros(T, length(b)))

function PolyaGamma{T, D}() where {T <: AbstractFloat, D}
    PolyaGamma(ones(T, D), zeros(T, D))
end

function Base.show(io::IO, pg::PolyaGamma)
    print(io, "$(typeof(pg))\n  b = $(pg.b)\n  c = $(pg.c)\b")
end

#######################################################################
# ExpFamilyDistribution interface

# The base measure of the PolyaGamma is an infinite sum with alterning
# signs. Since the base measure is not necessary while using VB
# training, we simply don't implement it.
function basemeasure(::PolyaGamma, X::AbstractMatrix)
    error("The base measure of the PolyaGamma is not implemented")
end


function gradlognorm(pdf::PolyaGamma{T, D}) where {T <: AbstractFloat, D}
    retval = (pdf.b ./ (2 * pdf.c)) .* tanh.(pdf.c ./ 2)

    # When c = 0 the mean is not defined but can be extended by
    # continuity by observing that lim_{x => 0} (e^(x) - 1) / x = 0
    # which lead to the mean = b / 4
    idxs = isnan.(retval)
    retval[idxs] .= pdf.b[idxs] / 4
    return retval
end

function lognorm(pdf::PolyaGamma{T, D}; perdim::Bool = false) where {T <: AbstractFloat, D}
    # cosh = (exp{2x} + 1) / (2 * exp{x})
    # logcosh = log(1 + exp{2x} - log(2) - x
    #sum(-pdf.b .* log.(cosh.(pdf.c ./ 2)))
    if ! perdim
        return sum(-pdf.b .* (log1pexp.(pdf.c) .- log(2) .- pdf.c ./ 2))
    else
        return -pdf.b .* (log1pexp.(pdf.c) .- log(2) .- pdf.c ./ 2)
    end
end

mean(pdf::PolyaGamma) = gradlognorm(pdf)

naturalparam(pdf::PolyaGamma) = - (pdf.c .^ 2) ./ 2

stats(::PolyaGamma, X::Matrix{<:AbstractFloat}) = X

stdparam(pdf::PolyaGamma) = (b=pdf.b, c=pdf.c)

function update!(pdf::PolyaGamma{T, D}, η::Vector) where {T <: AbstractFloat, D}
    c = sqrt.(-2 .* η)
    pdf.c[:] = c
    return nothing
end

