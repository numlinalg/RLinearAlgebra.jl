"""
    LinSysBlkRowRandCyclic <: LinSysBlkRowSampler

A mutable structure with fields to handle randomly permuted block sampling. Can allow for fixed
blocks or blocks whose entries are randomly permuted. After each cycle, a new random ordering is
created. If the last block would be smaller than the others, rows from a previous block are 
added to keep all blocks the same size.

# Fields
- `blockSize::Int64` - Specifies the size of each block.
- `constantBlock::Bool` - A variable specifying if the user would like for the rows to be randomly
permuted. 
- `nBlocks::Int64` - Variable that contains the number of blocks overall.
- `order::Vector{Int64}` - The order that the blocks will be used to generate updates.
- `blocks::Vector{Int64}` - The list of all the row in each block.

Calling `LinSysVecColBlockRandCyclic()` defaults to setting `blockSize` to 2 and `constantBlock` to true. The `sample`
function will handle the re-initialization of the fields once the system is provided.
"""
mutable struct LinSysBlkRowRandCyclic <: LinSysBlkRowSampler
    blockSize::Int64
    constantBlock::Bool
    nBlocks::Int64
    order::Vector{Int64}
    blocks::Vector{Int64}
end

LinSysBlkRowRandCyclic(blockSize, constantBlock) = LinSysBlkRowRandCyclic(blockSize, constantBlock, 1, Int64[], Int64[])
LinSysBlkRowRandCyclic(blockSize) = LinSysBlkRowRandCyclic(blockSize, true, 1, Int64[], Int64[])
LinSysBlkRowRandCyclic() = LinSysBlkRowRandCyclic(2, true, 1,  Int64[], Int64[])

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowRandCyclic,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    if iter == 1
        m, n = size(A)
        # Determine number of blocks
        type.nBlocks = div(m, type.blockSize) + (rem(m, type.blockSize) == 0 ? 0 : 1)
        blockIdxs = type.blockSize * type.nBlocks
        lastBlockStart = blockIdxs - type.blockSize + 1 
        type.blocks = Vector{Int64}(undef, blockIdxs)
        # Block definitions
        if type.constantBlock
            type.blocks[1:lastBlockStart - 1] .= collect(1:lastBlockStart - 1)
            if rem(n, type.blockSize) == 0
                type.blocks[lastBlockStart:blockIdxs] .= collect(lastBlockStart:m)
            else
                # maintain size of last block using last blockSize rows 
                type.blocks[lastBlockStart:blockIdxs] .= collect(vcat(m - type.blockSize + 1:lastBlockStart - 1, lastBlockStart:m))
            end
        
        else
            # If non-constant blocks randomly permute rows
            if rem(n, type.blockSize) == 0
                type.blocks .= randperm(m)
            else
                # maintain size of last block using last blockSize rows
                type.blocks[1:m] .= randperm(m)
                type.blocks[m:m + type.blockSize] .= type.blocks[type.blockSize]
            end

        end

        # Allocate the order the blocks will be sampled in
        type.order = randperm(type.nBlocks)
            
    end
    # So that iteration 1 corresponds to bIndx 1 use iter - 1 
    bIndx = rem(iter - 1, type.nBlocks) + 1
    # Reshuffle blocks
    if bIndx == 1
        type.order = randperm(type.nBlocks)
    end

    block = type.order[bIndx] 
    row_idx = type.blocks[type.blockSize * (block - 1) + 1:type.blockSize * block]
    SA = A[row_idx, :]
    Sb = b[row_idx]
    # Residual of the linear system
    res = SA * x - Sb

    return row_idx, SA, res
end
