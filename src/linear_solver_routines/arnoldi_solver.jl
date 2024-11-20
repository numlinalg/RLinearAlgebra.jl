# Date 09/10/2024
# Author: Christian Varner
# Purpose: Implement deterministic solver.

"""
    arnoldi!(x::AbstractVector, A::AbstractMatrix, b::AbstractVector, k::Int)

Finds an approximate solution to the linear system `Ax = b` by doing an Arnoldi Iteration
to build an orthonormal basis for the Krylov Subspace of size `k`. 
    
The method was detailed in

Timsit, Grigori, and Balabanov. "Randomized Orthogonal Projection Methods 
for Krylov Subspace Solvers." https://arxiv.org/abs/2302.07466.

# Parameters

- `x::AbstractVector`, initial solution to the linear system `Ax = b`. Will be overwritten
    with the solution found by the method.
- `A::AbstractMatrix`, coefficient matrix for linear system.
- `b::AbstractVector`, constant vector for linear system.
- `k::Int`, number of iterations to do (size of Krylov Subspace).

# Returns

- `H::AbstractMatrix`, upper hessenberg matrix formed during the iteration
- `V::AbstractMatrix`, orthogonal basis formed for the krylov subspace

Along with returning these matrices, `x` is also overwritten with the approximate solution.

"""
function arnoldi!(
    x::AbstractVector,
    A::AbstractMatrix,
    b::AbstractVector,
    k::Int
)

    # initialization of storage and initial residual
    r0 = b - A * x 
    beta = norm(r0)
    V = zeros(size(A)[1], k+1)
    H = zeros(k+1, k+1)

    V[:, 1] .= r0 ./ beta 
    for j in 1:k
        # next vector to add to basis
        z = view(V, :, j+1)
        mul!(z, A, view(V, :, j))

        # orthogonalization constants
        for i in 1:j
            v = view(V, :, i)
            H[i, j] = dot( v, z )
            z .-= H[i, j] .* v
        end
        
        # normalization
        H[j+1, j] = norm(z)
        z ./= H[j+1,j]
    end

    # solve linear system
    form_and_solve_hessenberg_system!(x, beta, H, V, k)
    return H, V
end
