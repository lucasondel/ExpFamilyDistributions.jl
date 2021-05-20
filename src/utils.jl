# SPDX-License-Identifier: MIT

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

    M = zeros(eltype(v), D, D)
    offset = 0
    for i in 1:D-1
        M[diagind(M, -i)] = v[offset+1:offset+D-i]
        offset += D-i
    end
    LowerTriangular(M)
end

"""
    matrix(diagM, trilM)

Returns a symmetrix matrix from the diagonal and the "tril" form of a
matrix.

See also [`vec_tril`](@ref), [`inv_vec_tril`](@ref)
"""
function matrix(diagM, trilM)
    T = eltype(diagM)
    utrilM = inv_vec_tril(trilM)
    Diagonal(diagM) + Symmetric(utrilM + utrilM')
end

@adjoint inv_vec_tril(M) = inv_vec_tril(M), Δ -> (vec_tril(Δ),)
@adjoint vec_tril(v) = vec_tril(v), Δ -> (inv_vec_tril(Δ),)
