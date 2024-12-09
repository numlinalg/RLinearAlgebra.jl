# Date: 12/02/2024
# Author: Christian Varner
# Purpose: Implement the randomized arnoldi iteration
# to build a sketched orthogonal basis for a krylov subspace

"""
    randomized_arnoldi(A::AbstractMatrix, q::AbstractVector, Omega::AbstractMatrix,
        k::Int64)

# Method

Computes a sketched orthonormal basis for the Krylov subspace
```math
\\text{span}(q, Aq, ..., A^{k-1}q),
```
and returns basis vectors that are approximately orthogonal, 
the sketched basis vectors, and an upper Hessenberg matrix containing the 
orthogonalizing coefficients for the sketched basis.

Let ``Q``, ``S``, and ``H`` be the matrices defined above, and let ``\\Omega`` be
the sketching matrix. A sketched orthonormal basis formed by the columns of ``S`` is
a orthonormal basis for the vector space defined by
```
\\text{span}(\\Omega q, \\Omega A q, \\Omega A^{k-1} q).
```

Furthermore, these matrix follow the following relationships.
```math
A Q_{k-1} = Q H,
```
and
```math
(\\Omega Q_{k-1})^{\\intercal} \\Omega A Q_{k-1} = H_{k-1},
```
where ``Q_{k-1}`` contains the first ``k-1`` columns of the matrix ``Q``, and ``H_{k-1}``
is the first ``k-1`` rows of the matrix ``H``.

# Reference(s)

Timsit, Grigori, Balabanov. "Randomized Orthogonal Projection Methods for Krylov Subspace
Solvers". arxiv, https://arxiv.org/pdf/2302.07466

# Arguments

- `A::AbstractMatrix`, matrix that is used to form Krylov subspace.
- `q::AbstractVector`, vector used to form krylov subspace. 
- `Omega::AbstractMatrix`, sketching matrix.
- `k::Int64`, how many vectors to add to basis.

# Returns 

- `Q::AbstractMatrix`, approximate basis vectors. Is of dimension `(size(q, 1), k)`.
- `S::AbstractMatrix`, sketched basis vectors. Is of dimension `(size(Omega, 1), k)`.
- `H::AbstractMatrix`, upper hessenberg matrix that stores the orthogonalizing 
coefficients for the sketched basis vectors. Is of dimension `(k, k - 1)`.
"""
function randomized_arnoldi(
        A::AbstractMatrix,     # matrix for krylov subspace
        q::AbstractVector,     # initial vector
        Omega::AbstractMatrix, # sketch matrix
        k::Int64)              # rank of krylov subspace
    
    # error checking
    @assert size(A, 1) == size(A, 2) "The matrix `A` is not square." 

    @assert size(A, 2) == size(q, 1)
    "Dimension of q is $(size(q, 1)), and the number of columns in `A` is"* 
    "$(size(A, 2)) which are not equal."
    
    @assert size(Omega, 2) == size(q, 1) 
    "Dimension of q is $(size(q, 1)), and the number of columns in `Omega` is"* 
    "$(size(Omega, 2)) which are not equal."
    
    @assert k >= 1 "Dimension requested is smaller than 1."
    @assert k <= size(Omega, 1) 
    "`k` is $(k) but Omega has $(size(Omega, 1)) rows, so a sketched orthogonal"* 
    "basis cannot be created."

    #initializations
    sz_full = size(q, 1)
    sz_sketch = size(Omega, 1)
    Q = zeros(sz_full, k)
    S = zeros(sz_sketch, k)
    H = zeros(k, k - 1)
    buffer = zeros(sz_full)

    # perform orthogonalization
    S[:, 1] .= Omega * q
    beta = norm(S[:, 1])
    
    S[:, 1] ./= beta
    Q[:, 1] .= q ./ beta
    for i in 2:k
        v = view(Q, :, i)
        mul!(v, A, view(Q, :, i - 1))

        s = view(S, :, i)
        mul!(s, Omega, v)

        # do gram schmidt
        sketched_basis = view(S, :, 1:(i-1))
        h = view(H, 1:(i-1), (i-1))
        mgs!(s, h, sketched_basis)

        # update q with approximate coefficients
        mul!(buffer, view(Q, :, 1:(i-1)), view(H, 1:(i-1), (i-1)))
        v .-= buffer
        
        # normalization
        mul!(s, Omega, v)
        H[i, (i-1)] = norm(s)
        v ./= H[i, (i-1)]
        s ./= H[i, (i-1)]
    end

    # return approximate basis, sketched basis, and upper hessenberg system
    return Q, S, H
end
