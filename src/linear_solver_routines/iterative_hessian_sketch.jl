# This file is part of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements rsubsolve function

"""
    IterativeHessianSketch <: LinSysSolveRoutine

A mutable structure that represents the Iterative Hessian Sketch Method.

See Mert Pilanci and Martin J. Wainwright. "Iterative Hessian Sketch: 
    Fast and Accurate Solution Approximation for Constrained Least-Squares."
    Journal of Machine Learning Research, 17(53), 1–38.
    http://jmlr.org/papers/v17/14-460.html

# Fields

- `A::AbstractArray`, the coefficient matrix for the linear system. Required as we need to compute the full residual.
- `b::AbstractVector`, the constant term (i.e., the linear system is `Ax = b`). Required as we need to compute the full residual.
- `step::Union{AbstractVector, Nothing}`, buffer array to hold solution of subproblem that provides update to x.
- `btilde::Union{AbstractVector, Nothing}`, buffer array to hold constant term in forming the subproblem to compute `step`.

# Constructors

        IterativeHessianSketch(A, b)

## Arguments

- `A::AbstractArray`, the coefficient matrix for the linear system. Required as we need to compute the full residual.
- `b::AbstractVector`, the constant term (i.e., the linear system is `Ax = b`). Required as we need to compute the full residual.
"""
mutable struct IterativeHessianSketch <: LinSysSolveRoutine 
    A::AbstractArray
    b::AbstractVector
    step::Union{AbstractVector, Nothing}
    btilde::Union{AbstractVector, Nothing}
end

function IterativeHessianSketch(A::AbstractArray, b::AbstractVector)
    return IterativeHessianSketch(A, b, nothing, nothing)
end

# Common rsubsolve interface for linear systems
function rsubsolve!(
    type::IterativeHessianSketch,
    x::AbstractVector,
    samp::Tuple{U,V,W} where {U<:AbstractArray, V<:AbstractArray, W<:AbstractVector},
    iter::Int64
)
    # samp[1] is the search direction
    # samp[2] is the sketched coefficient matrix
    # samp[3] is the residual of the sketched system

    # initialize buffer arrays
    m = size(samp[1])[1] # sketch size
    if iter == 1
        d = size(x)[1]
        type.step = Array{typeof(samp[2][1])}(undef, d) 
        type.btilde = Array{typeof(samp[2][1])}(undef, d)

        # sketch matrix will not have full column rank
        if m < d 
            @warn "The sampler's block_size might be too small for sensible inner problem solution. Algorithm will continue by solving the least squares problem instead (Caution: No theory)."
        end

    end

    # Form constant vector in sub-linear system 
    type.btilde .= m * ((type.A)' * (type.b - type.A * x))

    # Check if QR decomposition can be used on the sketched matrix SA to solve sub-linear system
    d = size(samp[2])[2] 
    if m >= d
        R = qr(samp[2]).R
        type.step .= R' \ type.btilde
        type.step .= R \ type.step
    else 
        # AS * SA might not full rank, apply More-Penrose inverse
        # Solving Q R R' Q x = b; b = type.btilde
        Q, R = qr(samp[2]')
        type.step .= Q' * type.btilde # R R' Q' x = Q' * b
        type.step .= R \ type.step    # Solve R z = Q' * b, z = R' Q' x
        type.step .= R' \ type.step   # Solve R' y = z, y = Q' x
        type.step .= Q * type.step    # Solve Q' * x = y
    end

    # update current iterate
    x .= x .+ type.step
    return nothing
end