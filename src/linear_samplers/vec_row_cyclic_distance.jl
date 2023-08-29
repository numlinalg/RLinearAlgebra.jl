# This file is part of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

"""
    LinSysVecRowDistCyclic <: LinSysVecRowSelect

A mutable structure with a field to store a cycling order. When the ordering 
is not specified, the ordering is filled in two steps. First, the distances 
between the current iterate and all hyperplanes as specified by the equations of
the system. Then, the ordering is the indices of these distances in decreasing 
order.

# Fields
- `order::Vector{Int64}`

Calling `LinSysVecRowDistCyclic()` defaults to setting `order` to an empty array. 
The `sample` function will handle the re-initialization of the fields once the 
system is provided.
"""
mutable struct LinSysVecRowDistCyclic <: LinSysVecRowSelect
    order::Vector{Int64}
end
LinSysVecRowDistCyclic() = LinSysVecRowDistCyclic(Int64[])

# Common sample interface for linear systems
function sample(
    type::LinSysVecRowDistCyclic,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    if isempty(type.order)
        type.order = sortperm(abs2.(A * x - b) ./ sum(A.^2, dims=2)[:], 
            rev=true)
    end 

    eqn_ind = popfirst!(type.order)

    return view(A, eqn_ind, :), view(b,eqn_ind)
end

#export LinSysVecRowRandCyclic