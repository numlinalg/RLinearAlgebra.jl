# This file is part of RLinearAlgebra.jl

"""
    LinSysBlkRowSparseSign <: LinSysBlkRowSampler

A mutable structure with fields to handle sparse sign row sketching where a sparse sign matrix
is multiplied by the matrix `A` from the left. More details of the methods are mentioned in the 
section 9.2 of 

Martinsson P-G, Tropp JA. "Randomized numerical linear algebra: Foundations and algorithms." 
Acta Numerica. 2020;29:403-572. doi:10.1017/S0962492920000021.


# Fields
- `block_size::Int64`, represents the embedding dimension of the sketch matrix.
- `numsigns::Int64`, storing how many signs we need to have for each column of the 
    sketch matrix, which can be chosen from `{2, 3, ..., block_size}`.
- `sketch_matrix::Union{AbstractMatrix, Nothing}`, buffer for storing the sparse sign sketching matrix.
- `scaling::Float64`, the standard deviation of the sketch, set to `sqrt(n / numsigns)`, calculated 
    only in the first iteration.
- `row_indices::Union{Vector{Int64}, Nothing}`, buffer for storing the sparse signs places 
    in each column.

# Constructors
- `LinSysBlkRowSparseSign()` defaults to setting `block_size` to `min(size(A, 1), 8)` and 
    `numsigns` to `min(block_size, 8)`.
"""
mutable struct LinSysBlkRowSparseSign <: LinSysBlkRowSampler
    block_size::Union{Int64, Nothing}
    numsigns::Union{Int64, Nothing}
    sketch_matrix::Union{AbstractMatrix, Nothing}
    scaling::Float64
    row_indices::Union{Vector{Int64}, Nothing}
    LinSysBlkRowSparseSign( block_size, 
                            numsigns, 
                            sketch_matrix, 
                            scaling,
                            row_indices,
                          ) = begin
        (block_size !== nothing) && @assert (block_size >= 2) "`block_size` must be greater than 1."
        (numsigns !== nothing) && @assert (numsigns >= 2) "`numsigns` must be greater than 1."
        return new(block_size, numsigns, sketch_matrix, scaling, row_indices)
    end
end

LinSysBlkRowSparseSign(;
                       block_size = nothing, 
                       numsigns = nothing,
                      ) = LinSysBlkRowSparseSign( block_size, 
                                                  numsigns, 
                                                  nothing, 
                                                  0.0,
                                                  nothing
                                                )

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowSparseSign,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    # Sketch matrix has dimension type.block_size (a pre-identified number of rows) by 
    # size(A,1) (matrix A's number of rows)
    if iter == 1
        # Set default value for block_size and check it is valid
        (type.block_size === nothing) && (type.block_size = min(size(A, 1), 8))
        (type.block_size > size(A,1)) && @warn "`block_size` should less than or"*
            " equal to row dimension, $(size(A,1))."

        # In default, we should sample min{type.block_size, 8} signs for each column.
        # Otherwise, we take an integer from 2 to type.block_size.
        (type.numsigns === nothing) && (type.numsigns = min(type.block_size, 8))
        @assert (type.numsigns <= type.block_size) "`numsigns` must less than the block size"*
            " of sketch matrix, $(type.block_size)."

        # Scaling value for sparse sign matrix
        type.scaling = sqrt(size(A,1) / type.numsigns)

        # Initialize the sparse sign matrix and assign values to iter
        type.sketch_matrix = zeros(Float64, type.block_size, size(A,1))

        # Initialize the row_indices to perform in-place allocation sampling
        type.row_indices = Vector{Int64}(undef, type.numsigns)
    end

    for col in axes(A, 1)
        # Choose `numsigns` random column indices
        sample!(1:type.block_size, type.row_indices, replace=false)
        
        # Fill the selected column indices with random -1 or 1
        type.sketch_matrix[:, col] .= 0  # Reset the entire row to 0
        type.sketch_matrix[type.row_indices, col] .= rand([-1, 1], type.numsigns)
    end

    # Scale the sparse sign matrix with dimensions
    type.sketch_matrix .*= type.scaling

    # Matrix after random sketch
    SA = type.sketch_matrix * A
    Sb = type.sketch_matrix * b

    # Residual
    res = SA * x - Sb
    
    # Output random sketch matrix, the matrix after random sketch, 
    # and the residual after random sketch
    return type.sketch_matrix, SA, res

end

# export LinSysBlkRowSparseSign