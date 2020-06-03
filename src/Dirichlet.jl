
using SpecialFunctions: loggamma, digamma


struct Dirichlet{T, D} <: ExpFamilyDistribution where T <: AbstractFloat
    α::Vector{T}

    function Dirichlet(α::Vector{T}) where T <: AbstractFloat
        new{T, length(α)}(α)
    end
end

Dirichlet{T, D}() where {T <: AbstractFloat, D} = Dirichlet(ones(T, D))

function Base.show(io::IO, d::Dirichlet)
    print(io, "$(typeof(d))\n")
    print(io, "  α = $(d.α)")
end

#######################################################################
# ExpFamilyDistribution interface

function basemeasure(pdf::Dirichlet, X::Matrix{T}) where T <: AbstractFloat
    zeros(T, size(X, 2))
end

function gradlognorm(pdf)
    α = stdparam(pdf)
    digamma.(α) .- digamma(sum(α))
end

function stats(::Dirichlet, X::Matrix{T}) where T <: AbstractFloat
    log.(X)
end

function lognorm(pdf::Dirichlet)
    α = stdparam(pdf)
    sum(loggamma.(α)) - loggamma(sum(α))
end

mean(pdf::Dirichlet) = pdf.α ./ sum(pdf.α)
naturalparam(pdf::Dirichlet) = pdf.α .- 1
stdparam(pdf::Dirichlet) = pdf.α

function update!(d::Dirichlet{T, D}, η::Vector{T}) where {T <: AbstractFloat, D}
    d.α[:] = η .+ 1
    return d
end

