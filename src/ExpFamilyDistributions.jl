# SPDX-License-Identifier: MIT

module ExpFamilyDistributions

#######################################################################
# Dependencies

using LinearAlgebra
using SpecialFunctions: loggamma, digamma
import Distributions
const Dists = Distributions
import Zygote: @adjoint, gradient
using PDMats

#######################################################################
# Utilities

export vec_tril
export inv_vec_tril
export matrix

include("utils.jl")

#######################################################################
# Parameter

export AbstractParameter
export naturalform
export realform
export jacobian
export reallocate

export DefaultParameter

include("parameter.jl")

#######################################################################
# ExpFamilyDistribution interface

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

export AbstractNormal
export Normal
include("normal.jl")

export AbstractNormalDiag
export NormalDiag
include("normaldiag.jl")

export AbstractGamma
export Gamma
include("gamma.jl")

export AbstractDirichlet
export Dirichlet
include("dirichlet.jl")

export AbstractWishart
export Wishart
include("wishart.jl")

end

