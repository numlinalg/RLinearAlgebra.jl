# Date: 09/04/2024
# Author: Christian Varner
# Purpose: Implement the randomized arnoldi solver from
# "Randomized Orthogonal Projection Methods for Krylov Subspace Solvers", 
# by Timsit, Grigori, Balabanov.

"""
    randomized_arnoldi!(x::AbstractVector, A::AbstractMatrix, b::AbstractVector, 
    k::Int64; sketch_matrix::Union{AbstractMatrix, Nothing})

Find an approximate solution of the linear system using the randomized arnoldi iteration.
This method was introduced in the following paper

Timsit, Grigori, and Balabanov. "Randomized Orthogonal Projection Methods for Krylov Subspace Solvers."
https://arxiv.org/abs/2302.07466.

# Parameters

- `x::AbstractVector`, initial solution to the linear system `Ax = b`. 
    Will be overwritten with the solution found by the method.
- `A::AbstractMatrix`, coefficient matrix for linear system.
- `b::AbstractVector`, constant vector for linear system.
- `k::Int`, number of iterations to do (size of Krylov Subspace).
- `sketch_matrix::Union{AbstractMatrix, Nothing}` (Optional; Default value `nothing`) 

# Returns

- `H::AbstractMatrix`, upper hessenberg matrix produced by the method
- `V::AbstractMatrix`, approximate orthogonal basis for the krylov subspace
- `S::AbstractMatrix`, sketched orthogonal basis for the sketched krylov subspace

Along with returning these matrices, `x` is also overwritten with the approximate solution.

"""
function randomized_arnoldi!(
    x::AbstractVector,                                      # initial guess - will be overwritten
    A::AbstractMatrix,                                      # coefficient matrix
    b::AbstractVector,                                      # constant vector
    k::Int;                                                 # max iteration limit 
    sketch_matrix::Union{AbstractMatrix, Nothing} = nothing # optional sketch matrix
)

    # check to make sure A is square
    if size(A)[1] != size(A)[2]
        throw(DomainError("Matrix A is not square."))
    end

    # check to see if sketch_matrix is initialized correctly; otherwise, create a gaussian matrix
    if isnothing(sketch_matrix)
        sketch_matrix = randn(k * cond(A), size(A)[1])
    elseif size(sketch_matrix)[1] < k
        @warn("Embedding space smaller then number of iterations. 
        Possible trouble with forming sketched basis. Caution advised.")
    end

    # initializations -- important quantities
    r0 = b - A * x                      
    sketch_r0 = sketch_matrix * r0
    beta = norm(sketch_r0)

    # initializations -- storage
    V = zeros(size(A)[1], k+1)              # Approximate basis for krylov space
    S = zeros(size(sketch_matrix)[1], k+1)  # Sketched basis for krylov space
    H = zeros(k+1, k+1)                     # Normalizing coefficients
    d = zeros(size(A)[1])                   # buffer array for making z approximately orthogonal
    s_prime = zeros(size(sketch_matrix)[1]) # buffer array for sketched basis vector

    # initial basis vectors
    V[:, 1] .= r0 ./ beta
    S[:, 1] .= sketch_r0 ./ beta

    # main loop
    for j in 1:k 
        # get vector to be add to the approximate basis
        z = view(V, :, j+1)
        mul!( z, A, view(V, :, j) )

        # get vector to add to the sketched basis 
        mul!(s_prime, sketch_matrix, z)             # p in the psuedo code

        # orthogonalizing constants
        for i in 1:j
            s = view(S, :, i)
            H[i, j] = dot(s, s_prime)
            s_prime .-= H[i, j] .* s
        end

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
    form_and_solve_hessenberg_system!(x, beta, H, V, k)
    return H, V, S
end