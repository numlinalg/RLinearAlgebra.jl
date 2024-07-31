# This file is part of RLinearAlgebra.jl
"""
    init_blocks_cyclic!(type::Union{LinSysBlkColRandCyclic,LinSysBlkRowRandCyclic}, dim::Int64)

This function intializes the `blockSize`, `n_blocks`, and `order` values for the `LinSysBlkColRandCyclic` 
and `LinSysBlkRowRandCyclic` data structures. If a set of blocks is already defined by the user then it 
warns the user which block indices are unused. If the blocks are not premade, it simply allocates blocks 
indices sequentially with a size that is floor(`n_blocks/dim`).
"""
function init_blocks_cyclic!(type::Union{LinSysBlkColRandCyclic,LinSysBlkRowRandCyclic}, dim::Int64)
        # If the user has not specified any blocks then allocate blocks sequentially 
        if typeof(type.blocks) <: Nothing
            @assert type.n_blocks > 0 "Number of blocks must be positive"
            if type.n_blocks > dim
                @warn "Setting `n_blocks` to the dimension. No obvious way to set blocks when `n_blocks` is greater than dimension. If you would like to do so, create a `Vector{Vector{Int64}}` with each sub vector containing indices of a block."
            end
            type.n_blocks = type.n_blocks <= dim ? type.n_blocks : dim
            bsize = div(dim, type.n_blocks)
            # If the remainder of the block size is zero then all blocks same size
            # otherwise last block is larger to include remaining entries with bsize
            remainder = rem(dim, bsize)
            last_bsize = remainder == 0 ? bsize : remainder + bsize 
            type.blocks = Vector{Vector{Int64}}(undef, type.n_blocks)
            #Allocate the block indices sequentially to all but last block which has different size
            for i in 1:(type.n_blocks-1)
                type.blocks[i] = collect((i - 1) * bsize + 1:i * bsize) 
            end

            # Place remaining entries in the last block
            type.blocks[type.n_blocks] = collect((type.n_blocks - 1) * bsize + 1:(type.n_blocks - 1) * bsize + last_bsize)   
        else
            @assert typeof(type.blocks) <: Vector{Vector{Int64}} "Your blocks must be entered as a 
            vector containing vectors of block indices represented as integers"
            type.n_blocks = size(type.blocks, 1)
            # Perform a check if all indices are used
            uniq_vals = Set(reduce(vcat, type.blocks))
            poss_vals = Set(1:dim)
            miss_vals = setdiff(poss_vals, uniq_vals)
            unused = !isempty(miss_vals)
            if unused
                @warn "Indices $miss_vals are unused in your blocks"
            end

        end

        # Allocate the order the blocks will be sampled in
        type.order = randperm(type.n_blocks)
        return nothing
end 
