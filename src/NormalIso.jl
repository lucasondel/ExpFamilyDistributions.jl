
mutable struct NormalIso{T, D} <: ExpFamilyDistribution where T <: AbstractFloat
    μ::Vector{T}
    σ²::T

    function NormalIso(μ::Vector{T}, σ²::Real) where T <: AbstractFloat
        new{T, length(μ)}(μ, T(σ²))
    end
end

function NormalIso(μ::Vector{T}) where T <: AbstractFloat
    D = length(μ)
    NormalIso(μ, T(1.))
end

function NormalIso{T, D}() where {T <: AbstractFloat, D}
    NormalIso(zeros(T, D), T(1))
end

function Base.show(io::IO, n::NormalIso)
    print(io, "$(typeof(n))\n")
    print(io, "  μ = $(n.μ)\n")
    print(io, "  σ² = $(n.σ²)\n")
end


#######################################################################
# ExpFamilyDistribution interface


function basemeasure(::NormalIso{T, D}, X::Matrix{T}) where {T <: AbstractFloat, D}
    retval = ones(T, size(X, 2))
    retval[:] .= -.5 * D * log(2 * pi)
    retval
end

function gradlognorm(pdf::NormalIso{T, D}) where {T, D}
    μ, σ² = stdparam(pdf)
    vcat(μ, D * σ² + dot(μ, μ))
end

function lognorm(pdf::NormalIso{T, D}) where {T, D}
    μ, σ² = stdparam(pdf)
    .5 * (D * log(σ²) + dot(μ, μ) / σ²)
end

mean(pdf::NormalIso) = pdf.μ

function naturalparam(pdf::NormalIso)
    μ, σ² = stdparam(pdf)
    λ = 1 / σ²
    vcat(λ * μ, -.5 * λ)
end

function stats(::NormalIso, X::Matrix{<:AbstractFloat})
    XᵀX = sum(X .* X, dims = 1)
    vcat(X, XᵀX)
end

stdparam(pdf::NormalIso) = (μ=pdf.μ, σ²=pdf.σ²)

function update!(pdf::NormalIso{T, D}, η::Vector{T}) where {T <: AbstractFloat, D}
    λμ = η[1:D]
    λ = -2 * η[D+1]
    pdf.σ² = 1/λ
    pdf.μ[:] = (1/λ) * λμ
    pdf
end

