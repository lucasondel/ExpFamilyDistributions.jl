module ExpFamilyDistributions

#######################################################################
# Dependencies
using LinearAlgebra
using SpecialFunctions: loggamma, digamma
using PDMats
import Distributions
const Dists = Distributions

#######################################################################
# ExpFamilyDistribution interface

export ExpFamilyDistribution
export δDistribution
export basemeasure
export gradlognorm
export kldiv
export lognorm
export loglikelihood
export mean
export naturalparam
export sample
export splitgrad
export stats
export stdparam
export update!

include("efdinterface.jl")

#######################################################################
# δ-Distributions

"""
    abstract type δDistribution

Supertype for the δ-distributions.
"""
abstract type δDistribution end

function Base.show(io::IO, ::MIME"text/plain", d::δDistribution)
    println(io, typeof(d), ":")
    print(io, "  μ = ", d.μ)
end

mean(d::δDistribution) = d.μ

sample(d::δDistribution, size=1) = [d.μ[1:end] for i in 1:size]

#######################################################################
# Distributions

export Normal
export δNormal
export NormalDiag
export δNormalDiag
export vec_tril
export inv_vec_tril

include("normal.jl")

export Gamma
export δGamma

include("gamma.jl")

export Dirichlet
export δDirichlet

include("dirichlet.jl")

export Wishart
export δWishart

include("wishart.jl")

end

