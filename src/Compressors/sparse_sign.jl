"""
   SparseSign <: Compressor

An implementation of the sparse sign compression method. This method forms a sparse matrix 
    with a fixed number of non-zeros per row or column depending on the direction that the 
    compressor is being applied. If compressing from the left, then the number of nonzeros 
    is fixed for each column. For compressing from the right, then the number of nonzeros is
    fixed for each row. Each nonzero entry takes a value of 
    ``\\{-1/\\sqrt{\\text{nnz}}, 1/\\sqrt{\\text{nnz}}\\}``[martinsson2020randomized](@cite).
    More concretely, if compressing from the left, we form ``m`` independent vectors 
    ``g_i \\in \\mathbb{R}^{s}`` with nnz nonzero entries which take a value of either 
    ``\\{-1/\\sqrt{\\text{nnz}}, 1/\\sqrt{\\text{nnz}}\\}``,  where ``m`` is the number of 
    rows in the matrix and ``s`` is the compression dimension. The final matrix will then be
    ```math 
    S=\\begin{bmatrix} g_1 g_2 \\hdots g_m \\end{bmatrix}
    ```
    .

# Fields
 - `cardinality::cardinality`, the direction the compression matrix is intended to be 
applied from.
 - `n_rows::Int64`, the number of rows in the compression matrix.
 - `n_cols::Int64`, the number of columns in the compression matrix.
 - `nnz::Int64`, the number of non-zero entries in each row if column compression or the 
 number of non-zero entries in each column if row compression.

# Constructor
Will be constructed with 
`SparseSign(;carinality = Left, compression_dim = 2, nnz::Int64 = 8)` where
- `carinality::Cardinality`, the direction `Left` or `Right` the compression matrix is 
intended to be applied from.
- `compression_dim`, the dimension of the vector/matrix after applying the compressor.
- `nnz::Int64`, the number of non-zeros per row/column in the sampling matrix. By default 
this is set to 8.
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
 - `cardinality::cardinality`, the direction the compression matrix is intended to be 
applied from.
 - `n_rows::Int64`, the number of rows in the compression matrix.
 - `n_cols::Int64`, the number of columns in the compression matrix.
 - `nnz::Int64`, the number of non-zero entries in each row if column compression or the 
 number of non-zero entries each column if row compression.
 - `scale::Vector{Number}`, the value of the non-zero entries.
 - `op::SparseMatrixCSC`, the SparseSign compressor matrix.

"""
mutable struct SparseSignRecipe <: CompressorRecipe
    cardinality::Type{<:Cardinality}
    n_rows::Int64
    n_cols::Int64
    nnz::Int64
    scale::Vector{Number}
    op::SparseMatrixCSC
end

function complete_compressor(ingredients::SparseSign, A::AbstractMatrix)
    if ingredients.cardinality == Left
        n_rows = ingredients.compression_dim
        n_cols = size(A, 1)
        initial_size = n_cols
    else
        n_rows = size(A, 2)
        n_cols = ingredients.compression_dim
        initial_size = n_rows
    end
    nnz = ingredients.nnz
    compression_dim = ingredients.compression_dim
    # get the element type of the matrix
    T = eltype(A)
    total_nnz = initial_size * nnz
    idxs = Vector{Int64}(undef, total_nnz)
    start = 1
    for i in 1:initial_size
        # every grouping of nnz entries corresponds to each row/column in sample
        stop = start + nnz - 1
        # Sample indices from the intial_size
        idx_view = view(idxs, start:stop)
        sample!(
            1:compression_dim, 
            idx_view, 
            replace = false, 
            ordered = true
        )
        start = stop + 1
    end
    
    # store the number in the type equivalent to the matrix A
    sc = convert(T, 1 / sqrt(nnz))
    # store as a vector to prevent reallocation during upadate
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

