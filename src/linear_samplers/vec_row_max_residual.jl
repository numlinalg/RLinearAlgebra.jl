# This file is pat of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

"""
    LinSysVecRowMaxResidual <: LinSysVecRowSelect

An immutable structure without fields that specifies choosing the 
linear equation in a system with the largest absolute residual at the current 
iterate.
"""
struct LinSysVecRowMaxResidual <: LinSysVecRowSelect end

# Common sample interface for linear systems
function sample(
    type::LinSysVecRowMaxResidual,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    eqn_indx = argmax(abs.(A * x - b))

    return A[eqn_indx,:], b[eqn_indx]
end
