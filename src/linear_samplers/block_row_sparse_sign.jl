# This file is part of RLinearAlgebra.jl

"""
    LinSysBlkRowSparseSign <: LinSysBlkRowSampler

A mutable structure with fields to handle sparse sign row sketching where a sparse sign matrix
is multiplied by the matrix `A` from the left. More details of the methods are mentioned in the 
section 9.2 of 

Martinsson P-G, Tropp JA. "Randomized numerical linear algebra: Foundations and algorithms." 
Acta Numerica. 2020;29:403-572. doi:10.1017/S0962492920000021.


# Fields
- `block_size::Int64`, represents the number of rows in the sketch matrix.
- `sparsity::Float64`, represents the sparsity of the sketch matrix. Suppose the sketch matrix 
    has dimensions of `block_size` rows and `n` columns; the sparsity, as described in the reference book, 
    can be chosen from `{2, 3, ..., block_size}`, representing the number of elements we want in each column.
- `sketch_matrix::Union{AbstractMatrix, Nothing}`, buffer for storing the sparse sign sketching matrix.
- `numsigns::Int64`, storing how many signs we need to have for each column of the 
    sketch matrix, calculated only in the first iteration by `max(floor(sparsity * size(A, 1)), 2)`.
- `scaling::Float64`, the standard deviation of the sketch, set to `sqrt(n / numsigns)`, calculated 
    only in the first iteration.

# Constructors
- `LinSysBlkRowSparseSign()` defaults to setting `block_size` to 8 and `sparsity` to `min(d, 8)`.
"""
mutable struct LinSysBlkRowSparseSign <: LinSysBlkRowSampler
    block_size::Int64
    function LinSysBlkRowSparseSign(block_size::Int64)
        @assert block_size > 0 "`block_size` must be positive."
        return new(block_size)
    end
    sparsity::Float64
    
    sketch_matrix::Union{Matrix{Int64}, Nothing}
    numsigns::Int64
    scaling::Float64
end

LinSysBlkRowSparseSign(;block_size, sparsity) = LinSysBlkRowSparseSign(block_size, sparsity,
    nothing, 0, 0.0)
LinSysBlkRowSparseSign(;block_size) = LinSysBlkRowSparseSign(block_size, -12345.0, nothing, 0, 
    0.0)
LinSysBlkRowSparseSign(;sparsity) = LinSysBlkRowSparseSign(8, sparsity, nothing, 0, 0.0)
LinSysBlkRowSparseSign() = LinSysBlkRowSparseSign(8, -12345.0, nothing, 0, 0.0)

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowSparseSign,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    # Sketch matrix has dimension type.block_size (a pre-identified number of rows) by 
    #size(A,1) (matrix A's number of rows)

    if iter == 1
        # @assert type.block_size > 0 "`block_size` must be positve."
        if type.block_size > size(A,1)
            @warn "`block_size` should be less than or equal to row dimension"
        end

        # In default, we should sample min{type.block_size, 8} signs for each column.
        # Otherwise, we take an integer from 2 to d with sparsity parameter.
        if type.sparsity == -12345.0
            type.numsigns = min(type.block_size, 8)
        elseif type.sparsity <= 0.0 || type.sparsity >= 1.0
            DomainError(sparsity, "Must be strictly between 0.0 and 1.0") |>
                throw
        else
            type.numsigns = max(floor(Int64, type.sparsity * size(A,1)), 2)
        end

        # Scaling value for saprse sign matrix
        type.scaling = sqrt(size(A,1) / type.numsigns)

        # Initialize the sparse sign matrix and assign values to iter
        type.sketch_matrix = zeros(Int64, type.block_size, size(A,1))
    end

    # Fill the sketch matrix with 1 and -1
    type.sketch_matrix = ifelse.(rand(type.block_size, size(A,1)) .> 0.5, 1, -1)  

    # Each column we want to have type.block_size - type.numsigns non-zero terms
    if type.block_size != type.numsigns
        for i in 1:size(A,1)
            row_perm = randperm(type.block_size)[1:(type.block_size - type.numsigns)]
            type.sketch_matrix[row_perm, i] = 0
        end
    end

    # Scale the sparse sign matrix with dimensions
    type.sketch_matrix .*=  type.scaling

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
