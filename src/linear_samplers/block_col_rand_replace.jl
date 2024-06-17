# This code was Written by Nathaniel Pritchard
"""
    LinSysVecBlkColReplace <: LinSysBlkColSampler 

A mutable structure with fields to handle randomly subset block sampling with replacement, also known as
vanilla block randomized kaczmarz. 

# Fields
- `blockSize::Int64` - Specifies the size of each block.
- `block::Vector{Int64}` - The list of all the rows in each block.

Calling `LinSysBlkColReplace()` defaults to setting `blockSize` to 2. The `sample`
function will handle the re-initialization of the fields once the system is provided.
"""
mutable struct LinSysBlkColReplace <: LinSysBlkColSampler 
    blockSize::Int64
    block::Vector{Int64}
end

LinSysBlkColReplace(blockSize) = LinSysBlkColReplace(blockSize, Int64[])
LinSysBlkColReplace() = LinSysBlkColReplace(2, Int64[])

# Common sample interface for linear systems
function sample(
    type::LinSysBlkColReplace,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    if iter == 1
        type.block = zeros(Int64, type.blockSize)
    end
    
    n = size(A, 2)
    #Sample the indices using the sample function from StatsBase
    type.block .= sample(1:n, type.blockSize, replace = false) 
    AS = A[:, type.block]
    res = A * x - b
    # Normal equation residual in the Sketched Block
    grad = AS' * (A * x - b)

    return type.block, AS, grad, res
end
