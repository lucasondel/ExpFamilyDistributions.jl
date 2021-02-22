module ExpFamilyDistributions

#######################################################################
# Dependencies
using LinearAlgebra
using SpecialFunctions: loggamma, digamma
import Distributions
const Dists = Distributions
import ForwardDiff
const FD = ForwardDiff
using PDMats

#######################################################################
# Utilities

export vec_tril
export inv_vec_tril
export matrix

include("utils.jl")

#######################################################################
# ExpFamilyDistribution interface

export Parameter
export naturalform
export realform

export Distribution
export basemeasure
export gradlognorm
export kldiv
export lognorm
export loglikelihood
export sample
export splitgrad
export stats
export stdparam

include("efdinterface.jl")

#######################################################################
# Distributions

export Normal
include("normal.jl")

export NormalDiag
include("normaldiag.jl")

export Gamma
include("gamma.jl")

export Dirichlet
include("dirichlet.jl")

export Wishart
include("wishart.jl")

end

