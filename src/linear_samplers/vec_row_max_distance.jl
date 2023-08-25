# This file is part of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

"""
    LinSysVecRowMaxDistance <: LinSysVecRowSelect

An immutable structure without fields that specifies choosing the 
linear equation in a system with the largest distance between the
current iterate and the hyperplane specified by the equation.
"""
struct LinSysVecRowMaxDistance <: LinSysVecRowSelect end

# Common sample interface for linear systems
function sample(
    type::LinSysVecRowMaxDistance,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    eqn_indx = argmax(abs.(A * x - b) ./ sum(A.^2, dims=2)[:])

    return A[eqn_indx,:], b[eqn_indx]
end