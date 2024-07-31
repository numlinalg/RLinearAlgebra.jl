# This code is part of RLinearAlgebra
"""
    LinSysBlkRowReplace <: LinSysBlkRowSampler 

A mutable structure with fields to store information for a sampling method that
forms a new block by uniformly sampling rows of `A` without replacement, 
also known as vanilla block randomized Kaczmarz. 

Necoara, Ion. “Faster Randomized Block Kaczmarz Algorithms.” SIAM J. Matrix Anal. Appl. 40 (2019): 1425-1452.

# Fields
- `block_size::Int64`, Specifies the size of each block.
- `block::Vector{Int64}`, The list of all the rows in each block.

# Constructors
Calling `LinSysBlkRowReplace()` defaults to setting `block_size` to 2. The `sample`
function will handle the re-initialization of the fields once the system is provided.
"""
mutable struct LinSysBlkRowReplace <: LinSysBlkRowSampler
    block_size::Int64
    block::Vector{Int64}
end

LinSysBlkRowReplace(;block_size = 2) = LinSysBlkRowReplace(
                                        block_size, 
                                        Int64[])

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
        @assert type.block_size > 0 "`block_size` must be positive"
        @assert type.block_size <= m "`block_size` must be less than row dimension"
        type.block = zeros(Int64, type.block_size)
    end

    #Sample the indices using the sample function from StatsBase
    type.block .= randperm(m)[1:type.block_size] 
    SA = A[type.block, :]
    Sb = b[type.block]
    # Residual of the linear system
    res = SA * x - Sb
    S = zeros(type.block_size, m)
    for i in 1:type.block_size
        S[i, type.block[i]] = 1
    end

    return S, SA, res
end
