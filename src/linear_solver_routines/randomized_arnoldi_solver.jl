# Date: 09/04/2024
# Author: Christian Varner
# Purpose: Implement the randomized arnoldi solver from
# "Randomized Orthogonal Projection Methods for Krylov Subspace Solvers", 
# by Timsit, Grigori, Balabanov.

"""
    randomized_arnoldi_solver!(x, A, b, k; sketch_matrix)

Find an approximate solution of the linear system using the randomized arnoldi iteration.
This method was introduced in the following paper

Timsit, Grigori, and Balabanov. "Randomized Orthogonal Projection Methods for Krylov Subspace Solvers."
https://arxiv.org/abs/2302.07466.

# Parameters
- `x::AbstractVector`, initial solution to the linear system `Ax = b`. Will be overwritten.
- `A::AbstractMatrix`, coefficient matrix for linear system.
- `b::AbstractVector`, constant vector for linear system.
- `k::Int`, number of iterations to do (size of Krylov Subspace).
- `sketch_matrix::Union{AbstractMatrix, Nothing}` (Optional; Default value `nothing`) 
"""
function randomized_arnoldi_solver!(
    x::AbstractVector,
    A::AbstractMatrix,
    b::AbstractVector,
    k::Int; # max iteration limit 
    sketch_matrix::Union{AbstractMatrix, Nothing} = nothing # should this just be a method
)

    # helper function to set up and solve the linear system
    function solve_linear_system!(
        x::AbstractVector, 
        beta::Float64, 
        H::AbstractMatrix, 
        V::AbstractMatrix, 
        k::Int64)

        ek = zeros(k)
        ek[1] = beta
        rho = H[1:k, 1:k] \ ek
        x .+= V[:, 1:k] * rho
    end

    # check to make sure A is square
    if size(A)[1] != size(A)[2]
        throw(DomainError("Matrix A is not square."))
    end

    # check to see if sketch_matrix is initialized
    if isnothing(sketch_matrix)
        # create a guassian matrix with size suggested by theory
        sketch_matrix = randn(k * cond(A), size(A)[1])
    end

    # initalizations
    r0         = b - A * x 
    sketch_r0  = sketch_matrix * r0
    beta       = norm(sketch_r0)

    # storage
    V = zeros(size(A)[1], k+1)
    S = zeros(size(sketch_matrix)[1], k+1)
    H = zeros(k+1, k+1)

    # initialization
    V[:, 1] = r0 / beta
    S[:, 1] = sketch_r0 / beta

    # main loop
    for j in 1:k
        z = A * V[:, j]
        p = sketch_matrix * z
        for i in 1:j
            H[i,j] = dot(S[:, i], p)
            p = p - H[i,j] * S[:, i]
        end
        z = z - V[:, 1:j] * H[1:j, j]
        s_prime = sketch_matrix * z
        H[j+1, j] = norm(s_prime)
        V[:, j+1] = z / H[j+1, j]
        S[:, j+1] = s_prime / H[j+1, j]
    end

    solve_linear_system!(x, beta, H, V, k)
    return H, V, S
end