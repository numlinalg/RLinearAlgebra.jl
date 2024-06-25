# This code is part of RLinearAlgebra.jl 
"""
    LinSysBlkRowRandCyclic <: LinSysBlkRowSampler

A mutable structure with fields to handle randomly permuted block sampling. User can input a list of
indices representing the blocks, otherwise the blocks will just be
created sequentially. After each cycle, a new random ordering is
created. If the last block would be smaller than the others, rows from a previous block are 
added to keep all blocks the same size.

# Fields
- `blockSize::Int64`, Specifies the size of each block.
- `nBlocks::Int64`, Variable that contains the number of blocks overall.
- `order::Union{Vector{Int64}, Nothing}`, The order that the blocks will be used to generate updates.
- `blocks::Union{Vector{Int64}, Nothing}`, The list of all the row in each block, will default to sequential order.
- `S::Union{Vector{Int64}, Nothing}`, The indices sampled from the current block.
- `SA::Union{AbstractMatrix, Nothing}`, The row block corresponding to this iteration's sampled indices.
- `res::Union{AbstractVector, Nothing}`, The row block residual.

Calling `LinSysVecRowBlockRandCyclic()` defaults to setting `blockSize` to 2 and `blocks` to sequential order. The `sample`
function will handle the re-initialization of the fields once the system is provided.

!!! Note:
For computational reasons all blocks will be forced to be the same size. This is done by determining how many rows are required for the last block to be the same size as the others and pulling that many rows from the second to last block.
"""
mutable struct LinSysBlkRowRandCyclic <: LinSysBlkRowSampler
    blockSize::Int64
    nBlocks::Int64
    blocks::Union{Vector{Int64}, Nothing}
    order::Union{Vector{Int64}, Nothing}
    S::Union{Vector{Int64}, Nothing}
    SA::Union{AbstractMatrix, Nothing}
    res::Union{AbstractVector, Nothing}
end

LinSysBlkRowRandCyclic(;blockSize=2, blocks=nothing) = LinSysBlkRowRandCyclic(blockSize, 1, blocks, nothing, nothing, nothing, nothing)

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowRandCyclic,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    if iter == 1
        m,n = size(A)
        init_blocks_cyclic!(type, m)
        # Allocate vector for block indices
        if typeof(type.S) <: Nothing
            type.S = Vector{Int64}(undef, type.blockSize)
        end

        # Allocate the blocks for storing sketched matrix
        if typeof(type.SA) <: Nothing
            T = eltype(A)
            type.SA = Matrix{T}(undef, type.blockSize, n)
        end

        # Allocate residual vector
        if typeof(type.res) <: Nothing
            T = eltype(A)
            type.res = Vector{T}(undef, type.blockSize)
        end
   
    end

    # So that iteration 1 corresponds to bIndx 1 use iter - 1 
    bIndx = rem(iter - 1, type.nBlocks) + 1
    # Reshuffle blocks
    if bIndx == 1
        randperm!(type.order)
    end

    block = type.order[bIndx] 
    copyto!(type.S, type.blocks[type.blockSize * (block - 1) + 1:type.blockSize * block])
    copyto!(type.SA, A[type.S, :])
    # residual initialized to b vector of current block
    copyto!(type.res, b[type.S])
    # Residual of the linear system
    mul!(type.res, type.SA, x, 1.0, -1.0)
    # return the sampler, the sketched matrix, the block residual
    return type, type.SA, type.res
end

# Implement mul functions that apply the sketching matrix to vectors and matrices
function *(Sampler::LinSysBlkRowRandCyclic, x::AbstractVector)
    return x[Sampler.S]
end

function *(Sampler::LinSysBlkRowRandCyclic, A::AbstractMatrix)
    return A[Sampler.S, :]
end

# Performs Sx = S * x
function LinearAlgebra.mul!(Sx::AbstractVector, Sampler::LinSysBlkRowRandCyclic, x::AbstractVector)
    # Copy the entries of y corresponding to the current block over to the vector x
    copyto!(Sx, x[Sampler.S])
end

# Performs SA = S * A
function LinearAlgebra.mul!(SA::AbstractMatrix, Sampler::LinSysBlkRowRandCyclic, A::AbstractMatrix)
    # Copy the entries of y corresponding to the current block over to the vector x
    copyto!(SA, A[Sampler.S, :])
end

# Performs Sb = β * Sb + α * S * x
function LinearAlgebra.mul!(Sb::AbstractVector, Sampler::LinSysBlkRowRandCyclic, x::AbstractVector, α::Number, β::Number)
    axpby!(α, x[Sampler.S], β, Sb)
end

# Performs SA = β * SA + α * S * A
function LinearAlgebra.mul!(SA::AbstractMatrix, Sampler::LinSysBlkRowRandCyclic, A::AbstractMatrix, α::Number, β::Number)
    axpby!(α, A[Sampler.S, :], β, SA)
end
