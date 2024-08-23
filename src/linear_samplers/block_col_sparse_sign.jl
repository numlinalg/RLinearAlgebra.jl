# This file is part of RLinearAlgebra.jl

"""
    LinSysBlkColSparseSign <: LinSysBlkColSampler

A mutable structure with fields to handle sparse sign column sketching where a sparse sign matrix
is multiplied by the matrix `A'` from the left. Methods are implemented as mentioned in section 9.2 
of "Martinsson P. G., Tropp J. A. Randomized numerical linear algebra: Foundations and algorithms." 
Acta Numerica, 2020, 29: 403-572.

# Fields
- `block_size::Int64`: Represents the number of rows in the sketch matrix.
- `sparsity::Float64`: Represents the sparsity of the sketch matrix. Suppose the sketch matrix 
  has dimensions of `d` rows and `n` columns; the sparsity, as described in the reference book, 
  can be chosen from `{2, 3, ..., d}`, representing the number of elements we want in each column.
- `numsigns::Int64`: Buffer for storing how many signs we need to have for each column of the 
  sketch matrix, calculated by `max(floor(sparsity * size(A, 1)), 2)`.
- `sketch_matrix::Union{AbstractMatrix, Nothing}`: Buffer for storing the Gaussian sketching matrix.
- `scaling::Float64`: The standard deviation of the sketch, set to `sqrt(n / numsigns)`.
- `rand_sign_matrix::Union{AbstractMatrix, Nothing}`: Buffer for storing the matrix used to build 
  the sketch matrix. It stores all possible signs for each iteration.
- `matrix_perm::Union{AbstractMatrix, Nothing}`: Buffer for storing the positions of all non-zero 
  entries in the sketch matrix.

# Constructors
- `LinSysBlkColSparseSign()` defaults to setting `block_size` to 8 and `sparsity` to `min(d, 8)`.
"""

mutable struct LinSysBlkColSparseSign <: LinSysBlkColSampler
    block_size::Int64
    sparsity::Float64
    numsigns::Int64
    sketch_matrix::Union{AbstractMatrix, Nothing}
    scaling::Float64
    rand_sign_matrix::Union{AbstractMatrix, Nothing}
    matrix_perm::Union{AbstractMatrix, Nothing}
end

LinSysBlkColSparseSign(block_size, sparsity) = LinSysBlkColSparseSign(block_size, sparsity, 0,
    nothing, 0.0, nothing, nothing)
LinSysBlkColSparseSign(block_size) = LinSysBlkColSparseSign(block_size, -12345.0, 0, nothing, 
    0.0, nothing, nothing)
LinSysBlkColSparseSign(;sparsity) = LinSysBlkColSparseSign(8, sparsity, 0, nothing, 0.0, 
    nothing, nothing)
LinSysBlkColSparseSign() = LinSysBlkColSparseSign(8, -12345.0, 0, nothing, 0.0, nothing, 
    nothing)

# Common sample interface for linear systems
function sample(
    type::LinSysBlkColSparseSign,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    # sketch matrix has dimension d (a pre-identified number of rows) by n (matrix A's 
    # number of Columns)
    d, n = type.block_size, size(A,2)

    if iter == 1
        @assert type.block_size > 0 "`block_size` must be positve."
        if type.block_size > n
            @warn "`block_size` should be less than or equal to column dimension"
        end

        # In default, we should sample min{d, 8} signs for each column.
        # Otherwise, we take an integer from 2 to d with sparsity parameter.
        if type.sparsity == -12345.0
            type.numsigns = min(d, 8)
        elseif type.sparsity <= 0.0 || type.sparsity >= 1.0
            DomainError(sparsity, "Must be strictly between 0.0 and 1.0") |>
                throw
        else
            type.numsigns = max(floor(Int64, type.sparsity * n), 2)
        end

        # Scaling value for saprse sign matrix
        type.scaling = sqrt(n / type.numsigns)

        # Initialize the sparse sign matrix and assign values to iter
        type.sketch_matrix = zeros(Int64, d, n)

        # Allocate for the rand_sign_matrix
        type.rand_sign_matrix = Matrix{Int64}(undef, d, n) 

        # Allocate for the permutation matrix
        type.matrix_perm = Matrix{Int64}(undef, type.numsigns, n) 
    end

    # Create a random matrix with 1 and -1
    type.rand_sign_matrix = ifelse.(rand(d, n) .> 0.5, 1, -1)  

    # Random permutation for choosing sparse signs
    type.matrix_perm =  sort(hcat([randperm(d) for _ in 1:n]...)[1:type.numsigns , :], dims = 1)
    # Make the position is suit for the whole matrix rand_sign_matrix
    type.matrix_perm .+= (d * (0:size(type.rand_sign_matrix, 2) - 1))'
    
    # Assign corresponding positions' value to S
    type.sketch_matrix[type.matrix_perm] = type.rand_sign_matrix[type.matrix_perm]
    # Scale the sparse sign matrix with dimensions
    type.sketch_matrix .*= type.scaling

    # Matrix after random sketch
    AS = A * type.sketch_matrix'

    # Residual of the linear system
    res = A * x - b

    # Normal equation residual in the Sketched Block
    grad = AS' * res
    
    return type.sketch_matrix', AS, grad, res

end

# export LinSysBlkColSparseSign
