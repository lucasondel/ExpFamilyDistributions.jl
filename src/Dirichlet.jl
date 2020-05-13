using LinearAlgebra

export Dirichlet
export DirichletFromStdParams

"""
    Dirichlet <: ExpFamilyDistribution

Dirichlet density. 

# Standard parameters:
  * ``α``: concentration values
  
# Natural parameters 
``η = \\begin{pmatrix} η_1 \\\\ \\text{vec}(η_2) \\end{pmatrix} 
    = \\begin{pmatrix} Λμ \\\\ \\text{vec}(Λ) \\end{pmatrix}``

# Sufficient statistics
``T(x) = \\begin{pmatrix} x \\\\ -\\frac{1}{2} \\text{vec}(xx^T)\\end{pmatrix}``

# Log-normalizer
``A(η) = \\frac{1}{2} η_1^Τ η_2^{-1} η_1 - \\frac{1}{2} \\ln |η_2|``

# Gradient of the log-normalizer
``∇Α(η) = \\begin{pmatrix} μ \\\\ Σ + μμ^Τ \\end{pmatrix}``

# Base measure
``B(x) = -\\frac{d}{2} \\ln(2π)``

where ``d`` is the dimension of ``x``.

# Constructors

    NormalFromStdParams(μ, [Σ=I])
    
Create a `Normal` density with mean `μ` and covariance 
matrix `Σ`. The covariance matrix should be symmetric positive 
definite. If `Σ` is omitted, the Normal is created assuming identity
covariance matrix. 
    
    NormalFromStdParams([T=Float64,] d)

Create a `d`-variate `Normal` density with zero mean and identity 
covariance matrix. 

# Examples

```jldoctest
julia> NormalFromStdParams([0., 0.], [1. 0.5; 0.5 1.]) 
Normal(η=[0.0, 0.0, 1.3333333333333333, -0.6666666666666666, -0.6666666666666666, 1.3333333333333333])

julia> NormalFromStdParams([0., 0.])
Normal(η=[0.0, 0.0, 1.0, 0.0, 0.0, 1.0])

julia> NormalFromStdParams(2)
Normal(η=[0.0, 0.0, 1.0, 0.0, 0.0, 1.0])

julia> NormalFromStdParams(Float32, 2)
Normal(η=Float32[0.0, 0.0, 1.0, 0.0, 0.0, 1.0])

```
"""
struct Normal <: ExpFamilyDistribution
    η::T where T<:AbstractVector
end


#######################################################################
# Helpers 

macro natparams(μ, Λ)
    return quote 
        local m = $(esc(μ))
        local L = $(esc(Λ))
        vcat(L * m, vec(L))
    end
end

macro splitnatparams(η)
    return quote 
        local l = length($(esc(η)))
        local d = convert(Int, (- 1 + sqrt(1 + 4 * l)) ÷ 2)
        view($(esc(η)), 1:d), Symmetric(reshape(view($(esc(η)), d+1:l), d, d))
    end
end

Base.show(io::IO, pdf::Normal) = println(io, "Normal(η=$(pdf.η))")


#######################################################################
# Constructors 

function NormalFromStdParams(
    μ::AbstractVector{<:AbstractFloat}, 
    Σ::AbstractMatrix{<:AbstractFloat}
)
    Λ = inv(Σ)
    Normal(@natparams μ Λ)
end

function NormalFromStdParams(μ::AbstractVector{<:AbstractFloat}) 
    d, T = length(μ), eltype(μ) 
    Λ = Matrix{T}(I, d, d) 
    Normal(@natparams μ Λ)
end

function NormalFromStdParams(T::Type{<:AbstractFloat}, d::Integer) 
    μ = zeros(T, d)
    Λ = Matrix{T}(I, d, d) 
    Normal(@natparams μ Λ)
end
NormalFromStdParams(d::Integer) = NormalFromStdParams(Float64, d)


#######################################################################
# ExpFamilyDistribution interface

function stats(::Normal, x::AbstractVector{<:AbstractFloat})
    vcat(x, -.5 * vec(x * x'))    
end

function lognorm(pdf::Normal)
    η1, η2 = @splitnatparams pdf.η
    .5 * (dot(η1, inv(η2) * η1) - logdet(η2))
end

function gradlognorm(pdf)
    μ, Σ = stdparams(pdf)
    vcat(μ, vec(Σ + μ * μ'))
end

function basemeasure(pdf::Normal, x::AbstractVector{<:AbstractFloat})
    -.5 * length(x) * log(2 * pi)
end

function stdparams(pdf::Normal)
    Λμ, Λ = @splitnatparams pdf.η
    Σ = inv(Λ)
    μ = Σ * Λμ 
    (μ=μ, Σ=Σ)
end

