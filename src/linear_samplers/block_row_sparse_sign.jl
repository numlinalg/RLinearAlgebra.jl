# This file is part of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

"""
    LinSysBlkRowSparseSign <: LinSysBlkRowSampler

A mutable structure with a field to represent different sparse sign embeddings. 

# Fields
- `sparsity::Float64`, sparsity represents the sparsity of the embedding matrix. 
Suppose the embedding matrix has dimension of d rows and n columns, the sparsity 
in the reference book can be chosen from {2, 3, ..., d} which represents how many elements
we want in each column.

- `embedding_dim::Int64`, embedding dimension represents the dimension of output. Note 
that, the output is residual, i.e. S*A*x - S*b.

Methods implemented as mentioned in section 9.2 of "Martinsson P G, Tropp J A. 
Randomized numerical linear algebra: Foundations and algorithms[J]. Acta Numerica, 
2020, 29: 403-572.".

# Constructors
Calling `LinSysBlkRowSparseSign()` defaults to set `sparsity` to min{d, 8}. 
"""

mutable struct LinSysBlkRowSparseSign <: LinSysBlkRowSampler
    block_size::Int64
    sparsity::Float64
    numsigns::Int64
    sketch_matrix::Union{AbstractMatrix, Nothing}
    scaling::Float64
    rand_sign_matrix::Union{AbstractMatrix, Nothing}
    matrix_perm::Union{AbstractMatrix, Nothing}
end

LinSysBlkRowSparseSign(block_size, sparsity) = LinSysBlkRowSparseSign(block_size, sparsity, 0,
    nothing, 0.0, nothing, nothing)
LinSysBlkRowSparseSign(block_size) = LinSysBlkRowSparseSign(block_size, -12345.0, 0, nothing, 
    0.0, nothing, nothing)
LinSysBlkRowSparseSign(;sparsity) = LinSysBlkRowSparseSign(8, sparsity, 0, nothing, 0.0, 
    nothing, nothing)
LinSysBlkRowSparseSign() = LinSysBlkRowSparseSign(8, -12345.0, 0, nothing, 0.0, nothing, 
    nothing)

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowSparseSign,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    # Embedding matrix has dimension d (a pre-identified number of rows) by n (matrix A's 
    # number of rows)
    d, n = type.block_size, size(A,1)

    if iter == 1
        @assert type.block_size > 0 "`block_size` must be positve."
        if type.block_size > n
            @warn "`block_size` should be less than row dimension"
        end

        # In default, we should sample min{d, 8} signs for each column.
        # Otherwise, we take an integer from 2 to d with sparsity parameter.
        if type.sparsity == -12345.0
            type.numsigns = min(size(A,1), 8)
        elseif type.sparsity <= 0.0 || type.sparsity >= 1.0
            DomainError(sparsity, "Must be strictly between 0.0 and 1.0") |>
                throw
        else
            type.numsigns = max(floor(Int64, type.sparsity * size(A,1)), 2)
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
    type.sketch_matrix .*=  type.scaling

    # Matrix after random embedding
    SA = type.sketch_matrix * A
    Sb = type.sketch_matrix * b

    # Residual
    res = SA * x - Sb
    
    # Output random embedding matrix, the matrix after random embedding, 
    # and the residual after random embedding
    return type.sketch_matrix, SA, res

end

# export LinSysBlkRowSparseSign
