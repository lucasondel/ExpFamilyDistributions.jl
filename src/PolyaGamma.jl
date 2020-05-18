
using StatsFuns

struct PolyaGamma{D} <: ExpFamilyDistribution
    b::AbstractVector
    c::AbstractVector

    function PolyaGamma(b::AbstractVector, c::AbstractVector)
        if size(b) ≠ size(c)
            error("Dimension mismathc: size(b) = $(size(b)) ≠ size(c) = $(size(c))")
        end
        new{length(b)}(b, c)
    end
end

PolyaGamma(b::AbstractVector) = PolyaGamma(b, zeros(eltype(b), length(b)))

function PolyaGamma{D}() where D
    PolyaGamma(ones(Float64, D), zeros(Float64, D))
end

function Base.show(io::IO, pg::PolyaGamma)
    print(io, "$(typeof(pg))\n  b = $(pg.b)\n  c = $(pg.c)\b")
end

#######################################################################
# ExpFamilyDistribution interface

# The base measure of the PolyaGamma is not zero but rather and
# infinite sum with alterning signs. Since the base measure is not
# necessary while using VB training, we simply set it to 0.
# WARNING:
#   Since, we do not implement the right base measure, the
#   log-likelihood of the function is only computed up to a constant.
function basemeasure(::PolyaGamma, X::AbstractMatrix)
    error("The base measure of the PolyaGamma is not implemented")
end


function gradlognorm(pdf::PolyaGamma)
    T = eltype(pdf.c)
    replace!((pdf.b ./ (2 * pdf.c)) .* tanh.(pdf.c ./ 2), NaN => T(1.))
end

function lognorm(pdf::PolyaGamma{D}; perdim::Bool = false) where D
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

stats(::PolyaGamma, X::AbstractMatrix) = X

stdparam(pdf::PolyaGamma) = (b=pdf.b, c=pdf.c)

function update!(pdf::PolyaGamma{D}, η::AbstractVector) where D
    c = sqrt.(-2 .* η)
    pdf.c[:] = c
    return nothing
end

