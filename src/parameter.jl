
#######################################################################
# Parameter interface.

"""
    abstract type AbstractParameter{T} end

Abstract type for parameters of a member of the exponential family.
"""
abstract type AbstractParameter{T} end

"""
    naturalform(param)

Returns the natural form of the parameter.

See also: [`realform`](@ref).
"""
naturalform(param::AbstractParameter)

"""
    realform(param)

Returns the vector of parameters as stored in `param`.

See also: [`naturalform`](@ref).
"""
realform(param::AbstractParameter)


"""
    jacobian(param)

Returns the Jacobian of ξ (the real form) w.r.t. the natural form η.
"""
jacobian(::AbstractParameter)

#######################################################################
# Default parameter: store the parameters in their natural form.

struct DefaultParameter{T} <: AbstractParameter{T}
    ξ::T
end

naturalform(p::DefaultParameter) = identity(p.ξ)
realform(p::DefaultParameter) = p.ξ
jacobian(p::DefaultParameter) = Diaognal(ones(eltype(p.ξ), length(p.ξ)))

