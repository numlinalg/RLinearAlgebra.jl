"""
    SparseSign <: Compressor

An implementation of the sparse sign compression method. This method forms a sparse matrix 
    with a fixed number of non-zeros per row or column depending on the direction that the 
    compressor is being applied. See Section 9.2 of [martinsson2020randomized](@cite) for 
    additional details.
    
# Mathematical Description
    
Let ``A`` be an ``m \\times n`` matrix that we want to compress. If we want 
    to compress ``A`` from the left (i.e., we reduce the number of rows), then 
    we create a sparse sign matrix, ``S``, with dimension ``s \\times m`` where 
    ``s`` is the compression dimension that is supplied by the user. 
    In this case, each column of ``S`` is generated independently by the following
    steps:
    
1. Randomly choose `nnz` components of the the ``s`` components of the column. Note, `nnz`
    is supplied by the user. 
2. For each selected component, randomly set it either to ``-1/\\sqrt{\\text{nnz}}`` or 
    ``1/\\sqrt{\\text{nnz}}`` with equal probability.
3. Set the remaining components of the column to zero. 

If ``A`` is compressed from the right, then we create a sparse sign matrix, ``S``,
    with dimension ``n \\times s``, where ``s`` is the compression dimension that 
    is supplied by the user. 
    In this case, each row of ``S`` is generated independently by the following steps: 

1. Randomly choose `nnz` coomponents fo the ``s`` components of the row. Note, `nnz`
    is supplied by the user.
2. For each selected component, randomly set it either to ``-1/\\sqrt{\\text{nnz}}`` or 
    ``1/\\sqrt{\\text{nnz}}`` with equal probability.
3. Set the remaining components of the row to zero. 

# Fields
- `cardinality::Type{<:Cardinality}`, the direction the compression matrix is intended to be 
    applied to a target matrix or operator. Values allowed are `Left` or `Right`.
- `compression_dim::Int64`, the target compression dimension. Referred to as ``s`` in the
    mathematical description. 
- `nnz::Int64`, the target number of nonzeros for each column or row of the spares sign 
    matrix.

!!! warn
    `nnz` must be no larger than `compression_dim`.

# Constructor
    
    SparseSign(;carinality=Left, compression_dim=2, nnz::Int64=8)

## Arguments 
- `carinality::Type{<:Cardinality}`, the direction the compression matrix is intended to be 
    applied to a target matrix or operator. Values allowed are `Left` or `Right`. By default 
    `Left` is chosen. 
- `compression_dim`, the target compression dimension. Referred to as ``s`` in the
    mathemtical description. By default this is set to 2.
- `nnz::Int64`, the number of non-zeros per row/column in the sampling matrix. By default 
    this is set to 8.

## Returns 
- A `SparseSign` object. 
"""
struct SparseSign <: Compressor
    cardinality::Type{<:Cardinality}
    compression_dim::Int64
    nnz::Int64
    # perform checks on the number of non-zeros
    SparseSign(cardinality, compression_dim, nnz) = begin
        if nnz > compression_dim 
            throw(DimensionMismatch("Number of non-zero indices, $nnz, must be less than\
            or equal to compression dimension, $compression_dim."))
        end
        return new(cardinality, compression_dim, nnz)
    end
end

function SparseSign(;cardinality = Left, compression_dim = 8, nnz::Int64 = 8)
    # Partially construct the sparse sign datatype
    return SparseSign(cardinality, compression_dim, nnz)
end

"""
    SparseSignRecipe <: CompressorRecipe

The recipe containing all allocations and information for the SparseSign compressor.

# Fields
- `cardinality::Type{<:Cardinality}`, the direction the compression matrix is intended to be 
    applied to a target matrix or operator. Values allowed are `Left` or `Right`.
- `n_rows::Int64`, the number of rows of the compression matrix.
- `n_cols::Int64`, the number of columns of the compression matrix.
- `nnz::Int64`, the number of non-zero entries in each row if `cardinality==Left` or the 
    number of non-zero entries each column if `cardinality==Right`.
- `scale::Vector{Number}`, the set of values of the non-zero entries of the Spares Sign
    compression matrix. 
- `op::SparseMatrixCSC`, the Spares Sign compression matrix.
"""
mutable struct SparseSignRecipe <: CompressorRecipe
    cardinality::Type{<:Cardinality}
    n_rows::Int64
    n_cols::Int64
    nnz::Int64
    scale::Vector{Number}
    op::SparseMatrixCSC
end

# This runs sparse sign assuming the keyword version had been previously run
function complete_compressor(ingredients::SparseSign, A::AbstractMatrix)
    a_rows, a_cols = size(A)
    # decide row and column dimensions based on inputted cardinality
    (n_rows, n_cols) = ingredients.cardinality == Left ? (ingredients.compression_dim, 
        a_rows) : (a_cols, ingredients.compression_dim)
    compression_dim = ingredients.compression_dim
    initial_size = ingredients.cardinality == Left ? a_rows : a_cols
    # get the element type of the matrix
    T = eltype(A)
    nnz = (ingredients.nnz == 8) ? min(8, compression_dim) : ingredients.nnz
    idxs = Vector{Int64}(undef, nnz * initial_size)
    start = 1
    for i in 1:initial_size
        # every grouping of nnz entries corresponds to each row/column in sample
        stop = start + nnz - 1
        # Sample indices from the intial_size
        @views sample!(
            1:compression_dim, 
            idxs[start:stop], 
            replace = false, 
            ordered = true
        )
        start = stop + 1
    end
    
    # store the number in the type equivalent to the matrix A
    sc = convert(T, 1 / sqrt(nnz))
    # store as a vector to prevent reallocation during upadate
    total_nnz = initial_size * nnz
    scale = [-sc, sc]
    signs = rand(scale, total_nnz)  
    ptr = collect(1:nnz:total_nnz+1)  

    #If left sketching store as a sparse matrix if right store as sparse matrix transpose
    if ingredients.cardinality != Left
        op = SparseMatrixCSC{T, Int64}(n_cols, n_rows, ptr, idxs, signs)'
    else
        op = SparseMatrixCSC{T, Int64}(n_rows, n_cols, ptr, idxs, signs)
    end

    return SparseSignRecipe(ingredients.cardinality, n_rows, n_cols, nnz, scale, op)
end

# This runs sparse Sign with both a matrix and vector input (for linear solver)
function complete_compressor(ingredients::SparseSign, A::AbstractMatrix, b::AbstractVector)
    complete_compressor(ingredients, A)
end

# allocations in this function are entirely due to bitrand call
function update_compressor!(S::SparseSignRecipe)
    # get compression dimension and initial size based on the cardinality if left just use 
    # size function which returns (n_rows, n_cols) otherwise reverse order
    (compression_dim, initial_size) = S.cardinality == Left ? size(S) : (S.n_cols, S.n_rows)
    start = 1
    n_col_ptr = length(S.op.colptr)
    for i in 1:n_col_ptr-1
        # every grouping of nnz entries corresponds to each row/column in sample
        stop = start + S.nnz - 1
        # Sample indices from the intial_size
        mv = view(S.op.rowval, start:stop)
        @views sample!(
            1:compression_dim, 
            mv, 
            replace = false, 
            ordered = true
        )
        start = stop + 1
    end

    # There is no inplace update of bitrand and using sample is slower
    rand!(S.op.nzval, S.scale)
    return 
end


# Do the right version
function mul!(
    x::AbstractVector, 
    S::SparseSignRecipe, 
    y::AbstractVector, 
    alpha::Number, 
    beta::Number
)
    # Check the compatability of the sizes of the things being multiplied
    vec_mul_dimcheck(x, S, y)
    mul!(x, S.op, y, alpha, beta)
    return
end

function mul!(
    x::AbstractVector, 
    S::CompressorAdjoint{SparseSignRecipe}, 
    y::AbstractVector, 
    alpha::Number, 
    beta::Number
)
    # Check the compatability of the sizes of the things being multiplied
    vec_mul_dimcheck(x, S, y)
    mul!(x, S.parent.op', y, alpha, beta)
    return
end

# Implement the matrix-Matrix Multiplication operators
# Begin with the left version
function mul!(
    C::AbstractMatrix, 
    S::SparseSignRecipe, 
    A::AbstractMatrix, 
    alpha::Number, 
    beta::Number
)
    left_mat_mul_dimcheck(C, S, A)
    mul!(C, S.op, A, alpha, beta)
end

# Now implement the right versions
function mul!(
    C::AbstractMatrix, 
    A::AbstractMatrix, 
    S::SparseSignRecipe, 
    alpha::Number, 
    beta::Number
)
    right_mat_mul_dimcheck(C, A, S)
    mul!(C, A, S.op, alpha, beta) 
    return
end

