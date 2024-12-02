# Date: 11/27/2024
# Author: Christian Varner
# Purpose: Implement the arnoldi iteration to build an orthogonal basis
# for a krylov subspace

"""
    arnoldi(A::AbstractMatrix, q::AbstractVector, k::Int64)

# Method

Compute a orthonormal basis for the Krylov subspace
```math
\\text{span}(q, Aq, ..., A^{k-1}q),
```
and returns the basis vectors and an upper Hessenberg matrix that contains
the coefficients for orthogonalizations. The hessenberg matrix, ``H``, 
and basis matrix, ``Q``, satisfy
```math
A Q_{k-1} = Q H,
```
and
```math
Q^\\intercal A Q = H.
```
where ``Q_{k-1}`` contains the first ``k-1`` columns of the matrix ``Q``.

# Reference(s)

TODO

# Arguments

- `A::AbstractMatrix`,
- `q::AbstractVector`,
- `k::Int64`,

# Returns

- `Q::AbstractMatrix`,
- `H::AbstractMatrix`,
"""
function arnoldi(A::AbstractMatrix, q::AbstractVector, k::Int64)
    
    # initializations
    sz = size(q, 1)
    Q = zeros(sz, k)
    H = zeros(k, k - 1)
    
    # perform orthogonalization
    Q[:, 1] .= q./norm(q)
    for i in 2:k

        # get new vector to orthogonalize
        q = view(Q, :, i)
        mul!(q, A, view(Q, :, i-1))
        
        # get coefficients
        basis = view(Q, :, 1:(i-1))
        h = view(H, 1:(i-1), (i-1))
        mgs!(q, h, basis)


        # normalize
        H[i, (i-1)] = norm(q)
        q ./= H[i, (i-1)]
    end

    return Q, H
end
