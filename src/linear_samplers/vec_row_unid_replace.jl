# This file is pat of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type
"""
    LinSysVecRowUnidSampler <: LinSysVecRowSampler

An immutable structure without fields that specifies randomly cycling from the rows of a
linear system with uniform probability and with replacement.
"""
struct LinSysVecRowUnidSampler <: LinSysVecRowSampler end

# Common sample interface for linear systems
function sample(
    type::LinSysVecRowUnidSampler,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    eqn_ind = rand(Base.OneTo(length(b)))

    return A[eqn_ind, :], b[eqn_ind]
end

#export LinSysVecRowUnidSampler
