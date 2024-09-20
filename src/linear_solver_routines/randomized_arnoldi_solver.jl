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
    x::AbstractVector,                                      # initial guess - will be overwritten
    A::AbstractMatrix,                                      # coefficient matrix
    b::AbstractVector,                                      # constant vector
    k::Int;                                                 # max iteration limit 
    sketch_matrix::Union{AbstractMatrix, Nothing} = nothing # optional sketch matrix
)

    # helper function to set up and solve linear system after main iteration
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

    # check to see if sketch_matrix is initialized correctly; otherwise, create a gaussian matrix
    if isnothing(sketch_matrix)
        sketch_matrix = randn(k * cond(A), size(A)[1])
    elseif size(sketch_matrix)[1] < k
        @warn("Embedding space smaller then number of iterations. Possible trouble with forming sketched basis. Caution advised.")
    end

    # initializations -- important quantities
    r0 = b - A * x                      
    sketch_r0 = sketch_matrix * r0
    beta = norm(sketch_r0)

    # initializations -- storage
    V = zeros(size(A)[1], k+1)              # Approximate basis for krylov space
    S = zeros(size(sketch_matrix)[1], k+1)  # Sketched basis for krylov space
    H = zeros(k+1, k+1)                     # Normalizing coefficients
    d = zeros(size(A)[1])                   # buffer array for making z orthogonal
    s_prime = zeros(size(sketch_matrix)[1]) # buffer array for sketched basis vector

    # initial basis vectors
    V[:, 1] .= r0 ./ beta
    S[:, 1] .= sketch_r0 ./ beta

    # main loop
    for j in 1:k 
        # get vector to be added to basis
        z = view(V, :, j+1)
        mul!( z, A, view(V, :, j) )

        # orthogonalizing constants
        mul!( s_prime, sketch_matrix, z )
        buffer = view(H, 1:j, j)
        mul!(buffer, view(S, :, 1:j)', s_prime)

        # orthogonalize (in sketch space) and update current basis
        mul!( d, view(V, :, 1:j), view(H, 1:j, j) )
        z .-= d
        
        # update our H, approximate basis V, and sketched basis S
        mul!(s_prime, sketch_matrix, z)
        H[j+1, j] = norm(s_prime)
        z ./= H[j+1, j]
        S[:, j+1] .= s_prime ./ H[j+1, j]
    end

    # solve the resulting linear system and return H, V, S for debugging purposes
    solve_linear_system!(x, beta, H, V, k)
    return H, V, S
end