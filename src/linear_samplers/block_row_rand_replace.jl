# This code was Written by Nathaniel Pritchard
"""
    LinSysBlkRowReplace <: LinSysBlkRowSampler 

A mutable structure with fields to handle randomly subset block sampling with replacement, also known as
vanilla block randomized kaczmarz. 

# Fields
- `block_size::Int64` - Specifies the size of each block.
- `block::Vector{Int64}` - The list of all the rows in each block.

Calling `LinSysBlkRowReplace()` defaults to setting `block_size` to 2. The `sample`
function will handle the re-initialization of the fields once the system is provided.
"""
mutable struct LinSysBlkRowReplace <: LinSysBlkRowSampler
    block_size::Int64
    block::Vector{Int64}
end

LinSysBlkRowReplace(block_size) = LinSysBlkRowReplace(block_size, Int64[])
LinSysBlkRowReplace() = LinSysBlkRowReplace(2, Int64[])

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowReplace,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    m = size(A,1)
    if iter == 1
        @assert type.block_size <= m "Block size must be less than row dimension"
        type.block = zeros(Int64, type.block_size)
    end

    #Sample the indices using the sample function from StatsBase
    type.block .= randperm(m)[1:type.block_size] 
    SA = A[type.block, :]
    Sb = b[type.block]
    # Residual of the linear system
    res = SA * x - Sb
    S = zeros(type.block_size, m)
    [S[i, type.block[i]] = 1 for i in 1:type.block_size]

    return S, SA, res
end
