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

- `A::AbstractArray` is the coefficient matrix for the linear system. Required as we need to compute the full residual.
- `y::AbstractVector` is the constant term (i.e., the linear system is Ax=y).
- `step::Union{AbstractVector, Nothing}` buffer array to hold solution of subproblem that provides update to x.
- `btilde::Union{AbstractVector, Nothing}` buffer array to hold constant term in forming the subproblem to compute `step`.
"""
mutable struct IterativeHessianSketch <: LinSysSolveRoutine 
    A::AbstractArray
    y::AbstractVector
    step::Union{AbstractVector, Nothing}
    btilde::Union{AbstractVector, Nothing}
end
# TODO: Default constructor?

# Common rsubsolve interface for linear systems
function rsubsolve!(
    type::IterativeHessianSketch,
    x::AbstractVector,
    samp::Tuple{U,V,W} where {U<:AbstractArray,V<:AbstractArray,W<:AbstractVector},
    iter::Int64
)
    # samp[1] is the search direction
    # samp[2] is the sketched coefficient matrix
    # samp[3] is the residual of the sketched system

    # initialize buffer arrays
    if iter == 1
        p = size(x)[1]
        type.step = Array{typeof(samp[2][1])}(undef, p) 
        type.btilde = Array{typeof(samp[2][1])}(undef, p) 
    end

    # form sub-linear system and solve 
    m = size(samp[1])[1]    
    type.btilde .= m .* ((type.A)'*(type.y - type.A*x))
    _,R = qr(samp[2])
    type.step .= R'\type.btilde
    type.step .= R\type.step

    # update current iterate
    x .= x .+ type.step

    return nothing
end