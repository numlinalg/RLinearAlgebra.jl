# This file is pat of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

# using Random

"""
    LinSysVecRowOneRandCyclic <: LinSysVecRowSelect

A mutable structure with a field to store a cycling order. Randomly specifies a cycling
order over the equations of a linear system. Once this ordering is specified, the ordering
is kept fixed.

# Fields
- `order::Union{Vector{Int64},Nothing}`

Calling `LinSysVecRowOneRandCyclic()` defaults to setting `order` to `nothing`. The `sample`
function will handle the re-initialization of the fields once the system is provided.
"""
mutable struct LinSysVecRowOneRandCyclic <: LinSysVecRowSelect
    order::Union{Vector{Int64},Nothing}
end
LinSysVecRowOneRandCyclic() = LinSysVecRowOneRandCyclic(nothing)

# Common sample interface for linear systems
function sample(
    type::LinSysVecRowOneRandCyclic,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    if iter == 1
        type.order = randperm(length(b))
    end

    eqn_ind = type.order[mod(iter, 1:length(b))]
    return A[eqn_ind, :], b[eqn_ind]
end

#export LinSysVecRowOneRandCyclic
