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
the coefficients for orthogonalization. The hessenberg matrix, ``H``, 
and basis matrix, ``Q``, satisfy
```math
A Q_{k-1} = Q H,
```
and
```math
Q_{k-1}^\\intercal A Q_{k-1} = H_{k-1}.
```
where ``Q_{k-1}`` contains the first ``k-1`` columns of the matrix ``Q``, and
``H_{k-1}`` contains the first ``k-1`` rows of the matrix ``H``.

# Reference(s)

Timsit, Grigori, Balabanov. "Randomized Orthogonal Projection Methods for Krylov
Subspace Solvers". arxiv, https://arxiv.org/pdf/2302.07466

Lloyd N. Trefethen and David Bau III. "Numerical Linear Algebra". SIAM, 1997, Lecture 33.

# Arguments

- `A::AbstractMatrix`, matrix used for the krylov subspace
- `q::AbstractVector`, vector used for the krylov subspace
- `k::Int64`, order of the krylov subspace. Number of vectors in krylov subspace.

# Returns

- `Q::Matrix{Float64}`, orthogonal basis vector for the krylov subspace. Is of dimension
`(size(q,1), k)`.
- `H::Matrix{Float64}`, upper hessenberg matrix containing orthogonalizing coefficients. Is
of dimension `(k, k - 1)`.
"""
function arnoldi(A::AbstractMatrix, q::AbstractVector, k::Int64)
    
    # error checking
    @assert size(A, 1) == size(A, 2) "The matrix `A` is not square."

    @assert size(A, 2) == size(q, 1) "Dimension of q is $(size(q, 1)), and the number of"*
    "columns in `A` is $(size(A, 2)) which are not equal."

    @assert k >= 1 "`k` is smaller than one."

    # initializations
    sz = size(q, 1)
    Q = zeros(sz, k)
    H = zeros(k, k - 1)
    
    # perform orthogonalization
    Q[:, 1] .= q./norm(q)
    for i in 2:k

        # get new vector to orthogonalize
        v = view(Q, :, i)
        mul!(v, A, view(Q, :, i-1))

        # get coefficients
        basis = view(Q, :, 1:(i-1))
        h = view(H, 1:(i-1), (i-1))
        mgs!(v, h, basis)

        # normalize
        H[i, (i-1)] = norm(v)
        v ./= H[i, (i-1)]
    end
        
    return Q, H
end
