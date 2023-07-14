# This file is pat of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

# using Random

"""
    LinSysVecColOneRandCyclic <: LinSysVecColSelect

A mutable structure with a field to store a cycling order. Randomly specifies a cycling
order over the equations of a linear system. Once this ordering is specified, the ordering
is kept fixed.

# Fields
- `order::Union{Vector{Int64},Nothing}`

Calling `LinSysVecColOneRandCyclic()` defaults to setting `order` to `nothing`. The `sample`
function will handle the re-initialization of the fields once the system is provided.
"""
mutable struct LinSysVecColOneRandCyclic <: LinSysVecColSelect
    order::Union{Vector{Int64},Nothing}
end
LinSysVecColOneRandCyclic() = LinSysVecColOneRandCyclic(nothing)

# Common sample interface for linear systems
function sample(
    type::LinSysVecColOneRandCyclic,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    if iter == 1
        type.order = randperm(length(x))
    end

    eqn_ind = type.order[mod(iter, 1:length(x))]
    return A[:, eqn_ind], x[eqn_ind]
end

#export LinSysVecColOneRandCyclic
