# This file is part of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

# using Random

"""
    LinSysVecRowRandCyclic <: LinSysVecRowSelect

A mutable structure with a field to store a cycling order. Randomly specifies a cycling
order the equations of a linear system. Once this ordering is exhausted by the solver,
a new random ordering is specified. This process is repeated

# Fields
- `order::Union{Vector{Int64},Nothing}`

Calling `LinSysVecRowOneRandCyclic()` defaults to setting `order` to `nothing`. The `sample`
function will handle the re-initialization of the fields once the system is provided.
"""
mutable struct LinSysVecRowRandCyclic <: LinSysVecRowSelect
    order::Union{Vector{Int64},Nothing}
end
LinSysVecRowRandCyclic() = LinSysVecRowRandCyclic(nothing)

# Common sample interface for linear systems
function sample(
    type::LinSysVecRowRandCyclic,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    mod_ind = mod(iter, 1:length(b))
    if mod_ind == 1
        type.order = randperm(length(b))
    end

    eqn_ind = type.order[mod_ind]
    return A[eqn_ind, :], b[eqn_ind]
end

#export LinSysVecRowRandCyclic
