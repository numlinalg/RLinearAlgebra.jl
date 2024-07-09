"""
    init_blocks_cyclic!(type::Union{LinSysBlkColRandCyclic,LinSysBlkRowRandCyclic}, dim)

This function intializes the `blockSize`, `n_blocks`, and `order` values for the `LinSysBlkColRandCyclic` and `LinSysBlkRowRandCyclic` data structures. If a set of blocks is already defined by the user then it checks it the vector specifying the blocks is the same as n_blocks * BlockSize, if it is not it appends the necessary number of indicies from the second to last block to the final block. If the blocks are not premade, it simply allocates blocks in sequential order.

"""
function init_blocks_cyclic!(type::Union{LinSysBlkColRandCyclic,LinSysBlkRowRandCyclic}, dim)
        @assert type.n_blocks < dim "The number of blocks can be no more than the dimension"
        # If the user has not specified any blocks then allocate blocks sequentially keeping all blocks the same size
        if typeof(type.blocks) <: Nothing
            bsize = div(dim, type.n_blocks)
            # If the remainder of the block size is zero then all blocks same size otherwise last block size remainder
            remainder = rem(dim, bsize)
            last_bsize = remainder == 0 ? bsize : remainder 
            type.blocks = Vector{Vector{Int64}}(undef, type.n_blocks)
            #Allocate the block indices sequentially
            for i in 1:(type.n_blocks-1)
                type.blocks[i] = collect((i - 1) * bsize + 1 : i * bsize) 
            end

            type.blocks[type.n_blocks] = collect((type.n_blocks - 1) * bsize + 1 : (type.n_blocks - 1) * bsize + last_bsize)   
        else
            @assert typeof(type.blocks) <: Vector{Vector{Int64}} "Your blocks must be entered as a 
            vector containing vectors of block indices represented as integers"
            type.n_blocks = size(type.blocks, 1)
            # Perform a check if all indices are used
            uniq_vals = unique(sort(reduce(vcat, type.blocks)))
            unused = false
            for i in 1:type.n_blocks
                # Check if each index is included in the vector
                unused = !(i in uniq_vals)
                if unused
                    @warn "Not all possible indices are used in these blocks!"
                    break
                end

            end

        end

        # Allocate the order the blocks will be sampled in
        type.order = randperm(type.n_blocks)
        return nothing
end 
