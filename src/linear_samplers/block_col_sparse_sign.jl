# This file is part of RLinearAlgebra.jl


"""
    LinSysBlkColSparseSign <: LinSysBlkColSampler

A mutable structure with fields to handle sparse sign column sketching where a sparse sign matrix
is multiplied by the matrix `A'` from the left. More details of the methods are mentioned in the 
section 9.2 of 

Martinsson P-G, Tropp JA. "Randomized numerical linear algebra: Foundations and algorithms." 
Acta Numerica. 2020;29:403-572. doi:10.1017/S0962492920000021.

# Fields
- `block_size::Int64`, represents the number of rows in the sketch matrix.
- `sketch_matrix::Union{AbstractMatrix, Nothing}`, buffer for storing the sparse sign sketching matrix.
- `numsigns::Int64`, buffer for storing how many signs we need to have for each column of the 
    sketch matrix, which can be chosen from `{2, 3, ..., block_size}.
- `scaling::Float64`, the standard deviation of the sketch, set to `sqrt(n / numsigns)`.

# Constructors
- `LinSysBlkColSparseSign()` defaults to setting `block_size` to 8 and `numsigns` to `min(d, 8)`.
"""
mutable struct LinSysBlkColSparseSign <: LinSysBlkColSampler
    block_size::Int64
    function LinSysBlkRowSparseSign(block_size::Int64)
        @assert block_size > 0 "`block_size` must be positive."
        return new(block_size)
    end
    # sparsity::Float64
    sketch_matrix::Union{Matrix{Int64}, Nothing}
    numsigns::Union{Int64, Nothing}
    scaling::Float64
end

LinSysBlkColSparseSign(;block_size, numsigns) = LinSysBlkColSparseSign(block_size, numsigns,
    nothing, 0, 0.0)
LinSysBlkColSparseSign(;block_size) = LinSysBlkColSparseSign(block_size, nothing, nothing, 0, 
    0.0)
LinSysBlkColSparseSign(;numsigns) = LinSysBlkColSparseSign(8, numsigns, nothing, 0, 0.0)
LinSysBlkColSparseSign() = LinSysBlkColSparseSign(8, nothing, nothing, 0, 0.0)

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
        # @assert type.block_size > 0 "`block_size` must be positve."
        if type.block_size > size(A,2)
            @warn "`block_size` should be less than or equal to column dimension"
        end

        # In default, we should sample min{type.block_size, 8} signs for each column.
        # Otherwise, we take an integer from 2 to type.block_size with sparsity parameter.
        if type.numsigns == nothing
            type.numsigns = min(type.block_size, 8)
        elseif type.numsigns <= 0 || type.numsigns >= size(A, 2)
            DomainError(type.numsigns, "Must be strictly between 0 and $(size(A, 2))") |>
                throw
        # else
        #     type.numsigns = max(floor(Int64, type.sparsity * size(A,2)), 2)
        end

        # Scaling value for saprse sign matrix
        type.scaling = sqrt(size(A,2) / type.numsigns)

        # Initialize the sparse sign matrix and assign values to iter
        type.sketch_matrix = zeros(Int64, type.block_size, size(A,2))

    end

    # Create a random matrix with 1 and -1
    type.sketch_matrix = ifelse.(rand(type.block_size, n) .> 0.5, 1, -1)  

    # Each column we want to have type.block_size - type.numsigns non-zero terms
    if type.block_size != type.numsigns
        for i in 1:size(A,2)
            row_perm = randperm(type.block_size)[1:(type.block_size - type.numsigns)]
            type.sketch_matrix[row_perm, i] = 0
        end
    end

    # Scale the sparse sign matrix with dimensions
    type.sketch_matrix .*=  type.scaling

    # Matrix after random sketch
    AS = A * type.sketch_matrix'

    # Residual of the linear system
    res = A * x - b

    # Normal equation residual in the Sketched Block
    grad = AS' * res
    
    return type.sketch_matrix', AS, grad, res

end

# export LinSysBlkColSparseSign
