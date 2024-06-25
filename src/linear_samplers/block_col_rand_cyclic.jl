# This code was written by Nathaniel Pritchard
"""
    LinSysBlkColRandCyclic <: LinSysBlkColSampler 

A mutable structure with fields to handle randomly permuted block sampling. Allows user to input an ordered
list of indices, which represents the blocks, if no list is provided the blocks are generated using 
sequential index ordering. After each cycle, a new random ordering is
created. If the last block would be smaller than the others, columns from a previous block are 
added to keep all blocks the same size.

# Fields
- `blockSize::Int64`, Specifies the size of each block.
- `nBlocks::Int64`, Variable that contains the number of blocks overall.
- `order::Vector{Int64}`, The order that the blocks will be used to generate updates.
- `blocks::Vector{Int64}`, The list of all columns indices ordered by block.
- `S::Union{Vector{Int64}, Nothing}`, The indicies in the sampled block.
- `AS::Union{AbstractMatrix, Nothing}`, The sampled columns of the matrix.
- `res::Union{AbstractVector, Nothing}`, The full residual.
- `grad::Union{AbstractVector, Nothing}`, The sketched gradient of the least squares problem.


Calling `LinSysBlkColRandCyclic()` defaults to setting `blockSize` to 2 and `blocks` to sequential order. The `sample`
function will handle the re-initialization of the fields once the system is provided.

!!! Note:
For computational reasons all blocks will be forced to be the same size. This is done by determining how many columns are required for the last block to be full and pulling that many columns from the second to last block.
"""
mutable struct LinSysBlkColRandCyclic <: LinSysBlkColSampler 
    blockSize::Int64
    nBlocks::Int64
    order::Union{Vector{Int64}, Nothing}
    blocks::Union{Vector{Int64}, Nothing}
    S::Union{Vector{Int64}, Nothing}
    AS::Union{AbstractMatrix, Nothing}
    res::Union{AbstractVector, Nothing}
    grad::Union{AbstractVector, Nothing}
end

LinSysBlkColRandCyclic(;blockSize=2, blocks=nothing) = LinSysBlkColRandCyclic(blockSize, 1, blocks, nothing, nothing, nothing, nothing, nothing)


# Common sample interface for linear systems
function sample(
    type::LinSysBlkColRandCyclic,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    if iter == 1
        m,n = size(A)
        init_blocks_cyclic!(type, n)
        # Allocate vector for block indices
        if typeof(type.S) <: Nothing
            type.S = Vector{Int64}(undef, type.blockSize)
        end

        # Allocate the blocks for storing sketched matrix
        if typeof(type.AS) <: Nothing
            T = eltype(A)
            type.AS = Matrix{T}(undef, m, type.blockSize)
        end

        # Allocate residual 
        if typeof(type.grad) <: Nothing
            T = eltype(A)
            type.res = A * x - b 
        end

        # Allocate block gradient vector
        if typeof(type.grad) <: Nothing
            T = eltype(A)
            type.grad = Vector{T}(undef, type.blockSize)
        end

    end
    # So that iteration 1 corresponds to bIndx 1 use iter - 1 
    bIndx = rem(iter - 1, type.nBlocks) + 1
    # Reshuffle blocks
    if bIndx == 1
        type.order = randperm(type.nBlocks)
    end

    block = type.order[bIndx] 
    copyto!(type.S, type.blocks[type.blockSize * (block - 1) + 1:type.blockSize * block])
    copyto!(type.AS, A[type.S, :])

    # Residual of the linear system
    mul!(type.res, A, x, -1, 1)  
    # Normal equation residual in the Sketched Block
    mul!(type.grad, transpose(type.AS), type.res, 1., 0.)

    return type, type.AS, type.grad, type.res
end


# Implement mul functions that apply the sketching matrix to vectors and matrices
function *(Sampler::LinSysBlkColRandCyclic, x::AbstractVector)
    @assert !(typeof(Sampler.AS) <: Nothing) "You need to call the sample function first"
    m,s = size(Sampler.AS)
    @assert size(x,1) == s "Your vector is not the same size as your sampling matrix"
    out = zeros(eltype(x), m)
    @views copyto!(out[Sampler.S], x)
    return out 
end

function *(A::AbstractMatrix, Sampler::LinSysBlkColRandCyclic)
    @assert !(typeof(Sampler.AS) <: Nothing) "You need to call the sample function first"
    n = size(A,2)
    @assert size(A,2) >= maximum(Sampler.blocks) "A does not have a large enough column dimension" 
    return A[:, Sampler.S] 
end

# Performs Sx = S * x
function LinearAlgebra.mul!(Sx::AbstractVector, Sampler::LinSysBlkColRandCyclic, x::AbstractVector)
    @assert !(typeof(Sampler.AS) <: Nothing) "You need to call the sample function first"
    m,s = size(Sampler.AS)
    @assert s == size(x,1) "Right-hand array, x, must match sampling dimension"
    @assert m == size(Sx,1) "Left-hand array, Sx, row dimension of block"
    # Copy the entries of y corresponding to the current block over to the vector x
    fill!(Sx, zero(eltype(Sx)))
    @views copyto!(Sx[Sampler.S], x)
end

# Performs SA = S * A
function LinearAlgebra.mul!(AS::AbstractMatrix, Sampler::LinSysBlkColRandCyclic, A::AbstractMatrix)
    @assert !(typeof(Sampler.AS) <: Nothing) "You need to call the sample function first"
    m,s = size(Sampler.AS)
    n = size(A,2)
    @assert s == size(AS, 2) "Column dimesion of left-hand matrix, AS, must match sampling dimension"
    @assert size(A,2) >= maximum(Sampler.blocks) "Right hand matrix, A, does not have a large enough column dimension" 
    # Copy the entries of y corresponding to the current block over to the vector x
    @views copyto!(AS, A[:, Sampler.S])
end

# Performs Sb = β * Sb + α * S * x and is used to update the solution vector in Least Squares solver
function LinearAlgebra.mul!(y::AbstractVector, Sampler::LinSysBlkColRandCyclic, x::AbstractVector, α::Number, β::Number)
    @assert !(typeof(Sampler.AS) <: Nothing) "You need to call the sample function first"
    m,s = size(Sampler.AS)
    @assert s == size(x,1) "Right-hand array, x, must match sampling dimension"
    @assert m == size(y,1) "Left-hand array, y, row dimension of block"
    axpby!(α, x, β, view(y, Sampler.S))
    #for i in 1:s
    #    temp = y[Sampler.S[i]] * β
    #    y[Sampler.S[i]] += x[i] * α
    #end

end

# Performs AS = β * AS + α * A * S
function LinearAlgebra.mul!(AS::AbstractMatrix, Sampler::LinSysBlkColRandCyclic, A::AbstractMatrix, α::Number, β::Number)
    @assert !(typeof(Sampler.AS) <: Nothing) "You need to call the sample function first"
    m,s = size(Sampler.AS)
    n = size(A,2)
    @assert s == size(AS, 2) "Column dimesion of left-hand matrix, AS, must match sampling dimension"
    @assert size(A,2) >= maximum(Sampler.blocks) "Right hand matrix, A, does not have a large enough column dimension" 
    axpby!(α, A[:, Sampler.S], β, AS)
end
