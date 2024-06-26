"""
    init_blocks_cyclic!(type::Union{LinSysBlkColRandCyclic,LinSysBlkRowRandCyclic}, dim)

This function intializes the `blockSize`, `nBlocks`, and `order` values for the `LinSysBlkColRandCyclic` and `LinSysBlkRowRandCyclic` data structures. If a set of blocks is already defined by the user then it checks it the vector specifying the blocks is the same as nBlocks * BlockSize, if it is not it appends the necessary number of indicies from the second to last block to the final block. If the blocks are not premade, it simply allocates blocks in sequential order.

"""
function init_blocks_cyclic!(type::Union{LinSysBlkColRandCyclic,LinSysBlkRowRandCyclic}, dim)
        @assert type.nBlocks < dim "The number of blocks can be no more than the dimension"
        # If the user has not specified any blocks then allocate blocks sequentially keeping all blocks the same size
        if typeof(type.blocks) <: Nothing
            bsize = div(dim, type.nBlocks)
            # If the remainder of the block size is zero then all blocks same size otherwise last block size remainder
            remainder = rem(dim, bsize)
            last_bsize = remainder == 0 ? bsize : remainder 
            type.blocks = Vector{Vector{Int64}}(undef, type.nBlocks)
            #Allocate the block indices sequentially
            for i in 1:(type.nBlocks-1)
                type.blocks[i] = collect((i - 1) * bsize + 1 : i * bsize) 
            end

            type.blocks[type.nBlocks] = collect((type.nBlocks - 1) * bsize + 1 : (type.nBlocks - 1) * bsize + last_bsize)   
        else
            @assert typeof(type.blocks) <: Vector{Vector{Int64}} "Your blocks must be entered as a 
            vector containing vectors of block indices"
            type.nBlocks = size(type.blocks, 1)
            @assert sum(size(type.blocks,1) .> dim) == 0 "Your blocks should be smaller than the dimension of the system"
        end

        # Allocate the order the blocks will be sampled in
        type.order = randperm(type.nBlocks)
end 
