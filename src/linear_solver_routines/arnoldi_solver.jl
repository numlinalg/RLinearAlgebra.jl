# Date 09/10/2024
# Author: Christian Varner
# Purpose: Implement deterministic solver.

"""
    arnoldi!(x, A, b, k)

Finds an approximate solution to the linear system `Ax = b` by doing an Arnoldi Iteration
to build an orthonormal basis for the Krylov Subspace of size `k`. The method was detailed
in

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

    r0 = b - A * x 
    beta = norm(r0)
    V = zeros(size(A)[1], k+1)
    V[:, 1] = r0 / beta

    H = zeros(k+1, k+1)

    for j in 1:k
        z = A * V[:, j]
        for i in 1:j
            H[i,j] = dot(V[:, i], z)
            z = z - H[i,j] * V[:, i]
        end
        #z = z - V[:, 1:j] * H[1:j, j]
        H[j+1,j] = norm(z)
        V[:,j+1] = z / H[j+1,j]
    end

    solve_linear_system!(x, beta, H, V, k)
    return H, V
end
