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
- `b::AbstractVector` is the constant term (i.e., the linear system is Ax=b).
- `step::Union{AbstractVector, Nothing}` buffer array to hold solution of subproblem that provides update to x.
- `btilde::Union{AbstractVector, Nothing}` buffer array to hold constant term in forming the subproblem to compute `step`.
"""
mutable struct IterativeHessianSketch <: LinSysSolveRoutine 
    A::AbstractArray
    b::AbstractVector
    step::Union{AbstractVector, Nothing}
    btilde::Union{AbstractVector, Nothing}
end

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
    m = size(samp[1])[1] # sketch size
    if iter == 1
        d = size(x)[1]
        type.step = Array{typeof(samp[2][1])}(undef, d) 
        type.btilde = Array{typeof(samp[2][1])}(undef, d)

        # sketch matrix will not have full column rank
        if m < d 
            @warn "Sketch matrix might be too small for a sensible inner problem solution!"
        end
    end

    # form sub-linear system and solve 
    type.btilde .= m .* ((type.A)'*(type.b - type.A*x))
    if m >= size(samp[2])[2] # nrow >= ncol for SA
        _,R = qr(samp[2])
        type.step .= R'\type.btilde
        type.step .= R\type.step
    else
        try
            LinearAlgebra.ldiv!(type.step, qr(samp[2]'*samp[2]), type.btilde)
        catch
            type.step .= zeros(size(x)[1])
            @warn "Sketch size might be too small. Encountered error in LinearAlgebra.ldiv!, no update applied!"
        end
    end

    # update current iterate
    x .= x .+ type.step

    return nothing
end