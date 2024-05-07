"""
    LinSysBlkRowReplace <: LinSysBlkRowSampler 

A mutable structure with fields to handle randomly subset block sampling with replacement, also known as
vanilla block randomized kaczmarz. 

# Fields
- `blockSize::Int64` - Specifies the size of each block.
- `block::Vector{Int64}` - The list of all the rows in each block.

Calling `LinSysBlkRowReplace()` defaults to setting `blockSize` to 2. The `sample`
function will handle the re-initialization of the fields once the system is provided.
"""
mutable struct LinSysBlkRowReplace <: LinSysBlkRowSampler
    blockSize::Int64
    block::Vector{Int64}
end

LinSysBlkRowReplace(blockSize) = LinSysBlkRowReplace(blockSize, Int64[])
LinSysBlkRowReplace() = LinSysBlkRowReplace(2, Int64[])

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowReplace,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    if iter == 1
        type.block = zeros(Int64, type.blockSize)
    end

    m = size(A,1)
    #Sample the indices using the sample function from StatsBase
    type.block .= sample(1:m, type.blockSize, replace = false) 
    SA = A[type.block, :]
    Sb = b[type.block]
    # Residual of the linear system
    res = SA * x - Sb

    return type.block, SA, res
end
