# This file is part of RLinearAlgebra.jl
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
- `order::Vector{Int64}`

Calling `LinSysVecColOneRandCyclic()` defaults to setting `order` to `nothing`. The `sample`
function will handle the re-initialization of the fields once the system is provided.
"""
mutable struct LinSysVecColOneRandCyclic <: LinSysVecColSelect
    order::Vector{Int64}
end
LinSysVecColOneRandCyclic() = LinSysVecColOneRandCyclic(Int64[])

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

    col_idx = type.order[mod(iter, 1:length(x))]

    # Search direction
    v = zeros(length(x))
    v[col_idx] = 1.0

    # Normal equation residual
    res = dot(A[:,col_ind], A * x - b)

    return v, A, res
end

#export LinSysVecColOneRandCyclic
