# This is file part of RLinearAlgebra.jl
"""
    LinSysBlkColRandCyclic <: LinSysBlkColSampler 

A mutable structure with fields to handle randomly permuted block sampling. 
After all blocks are called, a new random ordering is created. 

# Fields
- `n_blocks::Int64`, Variable that contains the number of blocks overall.
- `order::Union{Vector{Int64}, Nothing}`, The order that the blocks will be used 
    to generate updates.
- `blocks::Union{Vector{Vector{Int64}}, Nothing}`, The vector containing all the groupings 
    of column indices.
- `block_size::Union{Int64, Nothing}`, Variable that represents the smallest sketching 
    block size in iterations. Used in moving average method. 

# Constructors
Calling `LinSysBlkColRandCyclic()` defaults to setting `n_blocks` to 2 and `blocks` to be sequentially ordered. 
These values can be changed using the respective keyword arguments. 
The `sample` function will handle the re-initialization of the fields once the system is provided.
"""
mutable struct LinSysBlkColRandCyclic <: LinSysBlkColSampler 
    n_blocks::Int64
    order::Union{Vector{Int64}, Nothing}
    blocks::Union{Vector{Vector{Int64}}, Nothing}
    block_size::Union{Int64, Nothing}
end

LinSysBlkColRandCyclic(;
                       n_blocks = 2, 
                       blocks = nothing
                      ) = LinSysBlkColRandCyclic( n_blocks, 
                                                  nothing, 
                                                  blocks, 
                                                  nothing
                                                )

# Common sample interface for linear systems
function sample(
    type::LinSysBlkColRandCyclic,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    m, n = size(A)
    if iter == 1
        # Allocate space for blocks and intialize cycle
        init_blocks_cyclic!(type, n) 
        type.block_size = div(n, type.n_blocks)
    end
    # Use iter - 1 to ensure first iteration gives b_idx 1
    b_idx = rem(iter - 1, type.n_blocks) + 1
    # Reshuffle blocks
    if b_idx == 1
        type.order = randperm(type.n_blocks)
    end

    block = type.order[b_idx] 
    col_idx = type.blocks[block]
    AS = A[:, col_idx]

    # Residual of the linear system
    res = A * x - b
    # Normal equation residual in the Sketched Block
    grad = AS' * (A * x - b)
    
    bsize = size(col_idx, 1)
    # Create sketching matrix
    S = zeros(n, bsize)
    for i in 1:bsize
        S[col_idx[i], i] = 1
    end
    
    return S, AS, grad, res
end

