# This file is pat of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

# using Random

"""
    LinSysVecRowDetermCyclic <: LinSysVecRowSelect

An immutable structure without any fields. Specifies deterministic cycling through the
equations of a linear system.
"""
struct LinSysVecRowDetermCyclic <: LinSysVecRowSelect end

# Common sample interface for linear systems
function sample(
    type::LinSysVecRowDetermCyclic,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    eqn_ind = mod(iter, 1:length(b))
    return A[eqn_ind, :], b[eqn_ind]
end

#export LinSysVecRowDetermCyclic
