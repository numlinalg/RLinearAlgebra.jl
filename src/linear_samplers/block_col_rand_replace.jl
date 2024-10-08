# This file is part of RLinearAlgebra.jl
"""
    LinSysVecBlkColReplace <: LinSysBlkColSampler 

A mutable structure with fields to store information for a sampling method that
forms a new block by uniformly sampling columns of `A` without replacement.

# Fields
- `block_size::Int64`, specifies the size of each block.
- `block::Vector{Int64}`, the list of all the rows in each block.

# Constructors
Calling `LinSysBlkColReplace()` defaults to setting `block_size` to 2. The `sample`
function will handle the initialization of the fields once the system is provided.
"""
mutable struct LinSysBlkColReplace <: LinSysBlkColSampler 
    block_size::Int64
    block::Vector{Int64}
end

LinSysBlkColReplace(;block_size=2) = LinSysBlkColReplace(
                                        block_size, 
                                        Int64[])

# Common sample interface for linear systems
function sample(
    type::LinSysBlkColReplace,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    n = size(A, 2)
    if iter == 1
        @assert type.block_size > 0 "`block_size` must be positive"
        @assert type.block_size <= n "`block_size` must be less than column dimension"
        type.block = zeros(Int64, type.block_size)
    end
    
    #Sample the indices 
    type.block .= randperm(n)[1:type.block_size] 
    AS = A[:, type.block]
    res = A * x - b
    # Normal equation residual in the Sketched Block
    grad = AS' * (A * x - b)
    S = zeros(n, type.block_size)
    for i in 1:type.block_size
        S[type.block[i], i] = 1
    end

    return S, AS, grad, res
end
