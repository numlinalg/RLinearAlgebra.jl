# This file is part of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

"""
    LinSysVecRowResidCyclic <: LinSysVecRowSelect

A mutable structure with a field to store a cycling order. When the ordering 
is not specified, the ordering is filled by looking at the residuals at the
current iterate and ordering by decreasing residual. Once the order is exhausted
a new order is selected. 

# Fields
- `order::Vector{Int64}`

Calling `LinSysVecRowResidCyclic()` defaults to setting `order` to an empty array. 
The `sample` function will handle the re-initialization of the fields once the 
system is provided.
"""
mutable struct LinSysVecRowResidCyclic <: LinSysVecRowSelect
    order::Vector{Int64}
end
LinSysVecRowResidCyclic() = LinSysVecRowResidCyclic(Int64[])

# Common sample interface for linear systems
function sample(
    type::LinSysVecRowResidCyclic,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    if isempty(type.order)
        type.order = sortperm(abs.(A*x-b),rev=true)
    end 
    
    eqn_ind = popfirst!(type.order)

    return view(A, eqn_ind, :), view(b,eqn_ind)
end

#export LinSysVecRowResidCyclic