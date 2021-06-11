# This file is pat of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

# using Random

"""
    LinSysVecRowHopRandCyclic <: LinSysVecRowSelect

A mutable structure that specifies a cycling through the rows of a linear system, where the
cycling order is determined randomly once the current cycling order has been used `hop`
number of times. The solver randomly chooses the cycling order whenever necessary.

# Fields
- `order::Union{Vector{Int64},Nothing}`
- `hop::Int64`

# Constructors
- `LinSysVecRowHopRandCyclic()` defaults to setting the `order` to `nothing` and the `hop`
    to `5` (i.e., each ordering is used five times before sampling a new ordering).
- `LinSysVecRowHopRandCyclic(hop::Int64)` defaults to setting the `order` to `nothing` and
    the `hop` to whatever is specified by the argument.
"""
mutable struct LinSysVecRowHopRandCyclic <: LinSysVecRowSelect
    order::Union{Vector{Int64},Nothing}
    hop::Int64
end
LinSysVecRowHopRandCyclic() = LinSysVecRowHopRandCyclic(nothing, 5)
LinSysVecRowHopRandCyclic(hop::Int64) = LinSysVecRowHopRandCyclic(nothing, hop)

# Common sample interface for linear systems
function sample(
    type::LinSysVecRowHopRandCyclic,
    A::Matrix{Float64},
    b::Vector{Float64},
    x::Vector{Float64},
    iter::Int64
)

    # Determine whether the current ordering has been used `hop` times
    if mod(iter, 1:(length(b) * type.hop)) == 1
        type.order = randperm(length(b))
    end

    eqn_ind = type.order[mod(iter, 1:length(b))]
    return A[eqn_ind, :], b[eqn_ind]
end

#export LinSysVecRowHopRandCyclic
