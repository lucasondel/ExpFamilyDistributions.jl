# SPDX-License-Identifier: MIT

"""
    loggamma_dot

Wrapper around the log-gamma version to avoid conflicts with CUDA
"""
loggamma_dot(x::AbstractVector) = loggamma.(x)
loggamma_dot(x::CuArray) = lgamma.(x)

"""
    pdmat_logdet(M)

Log-determinant of a positive definite matrix.
"""
function pdmat_logdet(M::AbstractMatrix)
    U = cholesky(M).U
    sum(2 * log.(diag(U)))
end
pdmat_logdet(M::Diagonal) = sum(log.(diag(M)))

"""
    pdmat_inverse(M)

Inverse of a positive definite matrix.
"""
function pdmat_inverse(M::AbstractMatrix)
    U = cholesky(M).U
    I = typeof(M)(LinearAlgebra.I, size(M)...)
    U⁻ = I / U
    U⁻ * U⁻'
end
pdmat_inverse(M::Diagonal) = Diagonal(1 ./ diag(M))

"""
    vec_tril(M)

Returns the vectorized low-triangular part (diagonal not included) of
the matrix.

See also [`inv_vec_tril`](@ref), [`matrix`](@ref).
"""
function vec_tril(M)
    D, _ = size(M)
    vcat([diag(M, -i) for i in 1:D-1]...)
end

"""
    inv_vec_tril(v)

Returns a lower triangular matrix from a vectorized form. The diagonal
of the matrix is set to 0.

See also [`vec_tril`](@ref), [`matrix`](@ref)
"""
function inv_vec_tril(v)
    # Determine the dimension of the matrix from the length of the
    # vector. To do this, we solve the equation:
    #   x²/2 + x/2 - l = 0
    # The dimension is given by s (the positive solution) + 1.
    l = length(v)
    D = Int((-1 + sqrt(1 + 8*l))/2) + 1

    M = similar(v, D, D)
    fill!(M, 0)
    #M = zeros(eltype(v), D, D)
    offset = 0
    for i in 1:D-1
        M[diagind(M, -i)] = v[offset+1:offset+D-i]
        offset += D-i
    end
    M

    #LowerTriangular(M)
end

"""
    matrix(diagM, vec_trilM)

Returns a symmetrix matrix from the diagonal and the "tril" form of a
matrix.

See also [`vec_tril`](@ref), [`inv_vec_tril`](@ref)
"""
function matrix(diagM, vec_trilM)
    T = eltype(diagM)
    trilM = inv_vec_tril(vec_trilM)
    retval = trilM + trilM'
    retval[diagind(retval)] .= diagM
    retval
end
