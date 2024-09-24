# Date 09/10/2024
# Author: Christian Varner
# Purpose: Implement deterministic solver.

"""
    arnoldi!(x, A, b, k)

Finds an approximate solution to the linear system `Ax = b` by doing an Arnoldi Iteration
to build an orthonormal basis for the Krylov Subspace of size `k`. 
    
The method was detailed in

Timsit, Grigori, and Balabanov. "Randomized Orthogonal Projection Methods for Krylov Subspace Solvers."
https://arxiv.org/abs/2302.07466.

# Parameters
- `x::AbstractVector`, initial solution to the linear system `Ax = b`. Will be overwritten.
- `A::AbstractMatrix`, coefficient matrix for linear system.
- `b::AbstractVector`, constant vector for linear system.
- `k::Int`, number of iterations to do (size of Krylov Subspace).
"""
function arnoldi!(
    x::AbstractVector,
    A::AbstractMatrix,
    b::AbstractVector,
    k::Int
)

    # helper function to form and solve the linear system after main loop
    function solve_linear_system!(
        x::AbstractVector, 
        beta::Float64, 
        H::AbstractMatrix, 
        V::AbstractMatrix, 
        k::Int64)

        y = zeros(k)
        y[1] = beta
        z = H[1:k, 1:k] \ y
        x .+= V[:, 1:k] * z
    end

    # initialization of storage and initial residual
    r0 = b - A * x 
    beta = norm(r0)
    d = zeros(size(A)[1])
    V = zeros(size(A)[1], k+1)
    H = zeros(k+1, k+1)

    V[:, 1] .= r0 ./ beta 
    for j in 1:k
        # next vector to add to basis
        z = view(V, :, j+1)
        mul!(z, A, view(V, :, j))

        # orthogonalization constants
        buffer = view(H, 1:j, j)
        mul!(buffer, view(V, :, 1:j)', z)

        # orthogonalize and update basis
        mul!(d, view(V, :, 1:j), view(H, 1:j, j))
        z .-= d
        
        H[j+1,j] = norm(z)
        z ./= H[j+1,j]
    end

    # solve linear system
    #solve_linear_system!(x, beta, H, V, k)
    return H, V
end
