"""
    LinSysVecColBlockRandReplace <: LinSysVecColSelect

A mutable structure with fields to handle randomly subset block sampling with replacement, also known as
vanilla block randomized kaczmarz. 

# Fields
- `blockSize::Int64` - Specifies the size of each block.
- `block::Vector{Int64}` - The list of all the rows in each block.

Calling `LinSysVecColBlockRandReplace()` defaults to setting `blockSize` to 2. The `sample`
function will handle the re-initialization of the fields once the system is provided.
"""
mutable struct LinSysVecColBlockReplace <: LinSysVecColSelect
    blockSize::Int64
    block::Vector{Int64}
end

LinSysVecColBlockReplace(blockSize, constantBlock) = LinSysVecColBlockReplace(blockSize, Int64[])
LinSysVecColBlockReplace(blockSize) = LinSysVecColBlockReplace(blockSize, Int64[])
LinSysVecColBlockReplace() = LinSysVecColBlockReplace(2, Int64[])

# Common sample interface for linear systems
function sample(
    type::LinSysVecColBlockReplace,
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

#Function to update the solution 
function update_sol!(x::AbstractVector, update::AbstractVector, col_idx::Vector{Int64}, α::Real)
    x[col_idx] .-= α .* update
end
