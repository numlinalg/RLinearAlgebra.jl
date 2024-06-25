"""
    init_blocks_cyclic!(type::Union{LinSysBlkColRandCyclic,LinSysBlkRowRandCyclic}, dim)

This function intializes the `blockSize`, `nBlocks`, and `order` values for the `LinSysBlkColRandCyclic` and `LinSysBlkRowRandCyclic` data structures. If a set of blocks is already defined by the user then it checks it the vector specifying the blocks is the same as nBlocks * BlockSize, if it is not it appends the necessary number of indicies from the second to last block to the final block. If the blocks are not premade, it simply allocates blocks in sequential order.

"""
function init_blocks_cyclic!(type::Union{LinSysBlkColRandCyclic,LinSysBlkRowRandCyclic}, dim)
        
        @assert type.blockSize <= dim "Your block size cannot be larger than the dimension"
        # Determine number of blocks
        type.nBlocks = div(dim, type.blockSize) + (rem(dim, type.blockSize) == 0 ? 0 : 1)
        blockIdxs = type.blockSize * type.nBlocks
        lastBlockStart = blockIdxs - type.blockSize + 1 
        # If the user has not specified any blocks then allocate blocks sequentially keeping all blocks the same size
        if typeof(type.blocks) <: Nothing
            type.blocks = Vector{Int64}(undef, blockIdxs)
            type.blocks[1:lastBlockStart - 1] .= collect(1:lastBlockStart - 1)
            if rem(dim, type.blockSize) == 0
                type.blocks[lastBlockStart:blockIdxs] .= collect(lastBlockStart:dim)
            else
                # maintain size of last block using nencessary indices from second to last block 
                type.blocks[lastBlockStart:blockIdxs] .= collect(vcat(dim - type.blockSize + 1:lastBlockStart - 1, lastBlockStart:dim))
            end
        end
        b = size(type.blocks,1)
        @assert b <= dim "Your vector of indices representing blocks must be smaller than the dimension"
        if b < type.nBlocks * type.blockSize
            # Use indices from second to last block to keep the blocks the same size
            collect(type.blocks, dim - type.blockSize + 1:lastBlockStart - 1)
        end
        # Allocate the order the blocks will be sampled in
        type.order = randperm(type.nBlocks)
end 
