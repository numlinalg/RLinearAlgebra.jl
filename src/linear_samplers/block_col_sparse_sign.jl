# This file is part of RLinearAlgebra.jl


"""
    LinSysBlkColSparseSign <: LinSysBlkColSampler

A mutable structure with fields to handle sparse sign column sketching where a sparse sign matrix
is multiplied by the matrix `A` from the right. More details of the methods are mentioned in the 
section 9.2 of 

Martinsson P-G, Tropp JA. "Randomized numerical linear algebra: Foundations and algorithms." 
Acta Numerica. 2020;29:403-572. doi:10.1017/S0962492920000021.

# Fields
- `block_size::Int64`, represents the embedding dimension of the sketch matrix.
- `numsigns::Int64`, buffer for storing how many signs we need to have for each row of the 
    sketch matrix, which can be chosen from `{2, 3, ..., block_size}.
- `sketch_matrix::Union{AbstractMatrix, Nothing}`, buffer for storing the sparse sign sketching matrix.
- `scaling::Float64`, the standard deviation of the sketch, set to `sqrt(n / numsigns)`.

# Constructors
- `LinSysBlkColSparseSign()` defaults to setting `block_size` to 8 and `numsigns` to `min(block_size, 8)`.
"""
mutable struct LinSysBlkColSparseSign <: LinSysBlkColSampler
    block_size::Int64
    numsigns::Union{Int64, Nothing}
    sketch_matrix::Union{AbstractMatrix, Nothing}
    scaling::Float64
    LinSysBlkColSparseSign( block_size, 
                            numsigns, 
                            sketch_matrix, 
                            scaling,
                          ) = begin
        @assert (block_size > 0) "`block_size` must be positive."
        return new(block_size, numsigns, sketch_matrix, scaling)
    end
end

LinSysBlkColSparseSign(;
                       block_size = 8, 
                       numsigns = nothing,
                       ) = LinSysBlkColSparseSign( block_size, 
                                                   numsigns, 
                                                   nothing, 
                                                   0.0
                                                 )

# Common sample interface for linear systems
function sample(
    type::LinSysBlkColSparseSign,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    # Sketch matrix has dimension type.block_size (a pre-identified number of rows) by 
    # size(A,2) (matrix A's number of Columns)
    if iter == 1
        type.block_size > size(A, 2) && @warn "`block_size` should less than or \
                                                equal to column dimension, $(size(A,2))"

        # In default, we should sample min{type.block_size, 8} signs for each column.
        # Otherwise, we take an integer from 2 to type.block_size.
        type.numsigns == nothing && (type.numsigns = min(type.block_size, 8))
        type.numsigns <= 0 && @assert type.numsigns > 0 "`numsigns` must be positive."
        type.numsigns > 0 && @assert (type.numsigns <= type.block_size) "`numsigns` must less \
                                                                         than the block size \
                                                                         of sketch matrix, \
                                                                         $(type.block_size)."


        # Scaling value for saprse sign matrix
        type.scaling = sqrt(size(A,2) / type.numsigns)

        # Initialize the sparse sign matrix and assign values to iter
        type.sketch_matrix = zeros(Float64, size(A,2), type.block_size)

    end

    for row in 1:size(A,2)
        # Choose `numsigns` random column indices
        col_indices = randperm(type.block_size)[1:type.numsigns]
        
        # Fill the selected column indices with random -1 or 1
        type.sketch_matrix[row, :] .= 0  # Reset the entire row to 0
        type.sketch_matrix[row, col_indices] .= rand([-1, 1], type.numsigns)
    end


    # Scale the sparse sign matrix with dimensions
    type.sketch_matrix .*= type.scaling

    # Matrix after random sketch
    AS = A * type.sketch_matrix

    # Residual of the linear system
    res = A * x - b

    # Normal equation residual in the Sketched Block
    grad = AS' * res
    
    return type.sketch_matrix, AS, grad, res

end

# export LinSysBlkColSparseSign
