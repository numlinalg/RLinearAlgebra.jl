# This file is part of RLinearAlgebra.jl 
"""
    LinSysBlkRowRandCyclic <: LinSysBlkRowSampler

A mutable structure with fields to handle randomly permuted block sampling. After each cycle, 
a new random ordering is created. 

# Fields
- `n_blocks::Int64` - Variable that contains the number of blocks overall.
- `order::Union{Vector{Int64}, Nothing}` - The order that the blocks will be used to generate updates.
- `blocks::Union{Vector{Vector{Int64}}, Nothing}` - The list of all the row in each block.

Calling `LinSysVecColBlockRandCyclic()` defaults to setting `n_blocks` to 2.  The `sample`
function will handle the re-initialization of the fields once the system is provided.
"""
mutable struct LinSysBlkRowRandCyclic <: LinSysBlkRowSampler
    n_blocks::Int64
    order::Union{Vector{Int64}, Nothing}
    blocks::Union{Vector{Vector{Int64}}, Nothing}
end

LinSysBlkRowRandCyclic(;n_blocks=2, blocks = nothing) = LinSysBlkRowRandCyclic(n_blocks, nothing, blocks)

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowRandCyclic,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    m, n = size(A)
    if iter == 1
        init_blocks_cyclic!(type, m) 
    end

    # So that iteration 1 corresponds to bIndx 1 use iter - 1 
    bIndx = rem(iter - 1, type.n_blocks) + 1
    # Reshuffle blocks
    if bIndx == 1
        type.order = randperm(type.n_blocks)
    end

    block = type.order[bIndx] 
    row_idx = type.blocks[block]
    SA = A[row_idx, :]
    Sb = b[row_idx]
    bsize = size(row_idx,1)
    # Residual of the linear system
    res = SA * x - Sb
    # Define sketching matrix
    S = zeros(bsize, m)
    [S[i, row_idx[i]] = 1 for i in 1:bsize]

    return S, SA, res
end
