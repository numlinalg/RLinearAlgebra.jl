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

 1. Randomly choose `nnz` components fo the ``s`` components of the row. Note, `nnz`
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

  - `carinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left` or `Right`. By default
    `Left` is chosen.
  - `compression_dim`, the target compression dimension. Referred to as ``s`` in the
    mathemtical description. By default this is set to 2.
  - `nnz::Int64`, the number of non-zeros per row/column in the sampling matrix. By default
    this is set to min(compressiond_dim, 8).
  - `type::Type{<:Number}`, the type of elements in the compressor.
## Returns

  - A `SparseSign` object.
"""
struct SparseSign <: Compressor
    cardinality::Cardinality
    compression_dim::Int64
    nnz::Int64
    type::Type{<:Number}
    # perform checks on the number of non-zeros
    function SparseSign(cardinality, compression_dim, nnz, type)
        # the compression dimension must be positive and larger than the number of 
        # nonzeros
        if compression_dim <= 0
            throw(ArgumentError("Field `compression_dim` must be positive."))
        elseif nnz > compression_dim
            throw(ArgumentError("Number of non-zero indices, $nnz, must be less than \
            or equal to compression dimension, $compression_dim."))
        elseif nnz < 0
            throw(ArgumentError("Field `nnz` must be positive."))
        end

        return new(cardinality, compression_dim, nnz, type)
    end
end

function SparseSign(;
    cardinality=Left(), 
    compression_dim::Int64=2, 
    nnz::Int64=min(8, compression_dim),
    type::Type{N}=Float64
) where N<:Number
    # Partially construct the sparse sign datatype
    return SparseSign(cardinality, compression_dim, nnz, type)
end

"""
    SparseSignRecipe <: CompressorRecipe

The recipe containing all allocations and information for the SparseSign compressor.

# Fields

  - `cardinality::Cardinality`, the direction the compression matrix is intended to 
    be applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
  - `n_rows::Int64`, the number of rows of the compression matrix.
  - `n_cols::Int64`, the number of columns of the compression matrix.
  - `nnz::Int64`, the number of non-zero entries in each row if `cardinality==Left` or the
    number of non-zero entries each column if `cardinality==Right`.
  - `scale::Vector{Number}`, the set of values of the non-zero entries of the Spares Sign
    compression matrix.
  - `op::SparseMatrixCSC`, the Spares Sign compression matrix.
"""
mutable struct SparseSignRecipe <: CompressorRecipe
    cardinality::Cardinality
    n_rows::Int64
    n_cols::Int64
    nnz::Int64
    scale::Vector{Number}
    op::Union{SparseMatrixCSC, Adjoint{T, SparseMatrixCSC{T, I}}} where 
        {T<:Number, I<:Integer}
end

function complete_compressor(
    ingredients::SparseSign, A::AbstractMatrix
)
    # differentiate the number of rows and columns in the compressor based on the 
    # whether we intend to apply if from the left or right
    if typeof(ingredients.cardinality) == Left
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
    T = ingredients.type
    total_nnz = initial_size * nnz
    idxs = Vector{Int64}(undef, total_nnz)
    first_idx = 1
    # For every pair of column pointers you need to allocate row entries
    for i in 1:initial_size
        # every grouping of nnz entries corresponds to each row/column in sample
        last_idx = first_idx + nnz - 1
        # Sample indices from the intial_size
        idx_view = view(idxs, first_idx:last_idx)
        sample!(1:compression_dim, idx_view; replace=false, ordered=true)
        first_idx = last_idx + 1
    end

    # store the number in the type equivalent to the matrix A
    sc = convert(T, 1 / sqrt(nnz))
    # store as a vector to prevent reallocation during update
    scale = [-sc, sc]
    signs = rand(scale, total_nnz)
    # Allocatet the column pointers
    col_ptr = collect(1:nnz:(total_nnz + 1))

    # If left sketching store as a sparse matrix if right store as sparse matrix adjoint 
    # Use `SparseMatrixCSC` rather than `sparse` to ensure correct dimensions of the sparse 
    # matrix. Sparse has at times created matrices with too few rows/columns
    if typeof(ingredients.cardinality) == Left
        op = SparseMatrixCSC{T,Int64}(n_rows, n_cols, col_ptr, idxs, signs)
    else
        op = adjoint(SparseMatrixCSC{T,Int64}(n_cols, n_rows, col_ptr, idxs, signs))
    end

    return SparseSignRecipe(ingredients.cardinality, n_rows, n_cols, nnz, scale, op)
end

# allocations in this function are entirely due to bitrand call
function update_compressor!(S::SparseSignRecipe)
    # get compression dimension and initial size based on the cardinality if left just use 
    # size function which returns (n_rows, n_cols) otherwise reverse order
    cardinality = typeof(S.cardinality)
    (compression_dim, initial_size) = cardinality <: Left ? size(S) : (S.n_cols, S.n_rows)
    #alias whether the sparse matrix is transposed
    op = cardinality <: Left ? S.op : S.op.parent
    first_idx = 1
    n_col_ptr = length(op.colptr)
    # For every pair of column pointers you need to allocate row entries
    for i in 1:(n_col_ptr - 1)
        # every grouping of nnz entries corresponds to each row/column in sample
        last_idx = first_idx + S.nnz - 1
        # Fill in new indices to update the compressor 
        #mv = view(op.rowval, first_idx:last_idx)
        sample!(1:compression_dim, view(op.rowval, first_idx:last_idx), replace=false, ordered=true)
        first_idx = last_idx + 1
    end

    # There is no inplace update of bitrand and using sample is slower
    rand!(op.nzval, S.scale)
    return nothing
end

# Do the right version
function mul!(
    x::AbstractVector, S::SparseSignRecipe, y::AbstractVector, alpha::Number, beta::Number
)
    # Check the compatability of the sizes of the things being multiplied
    vec_mul_dimcheck(x, S, y)
    mul!(x, S.op, y, alpha, beta)
    return nothing
end

function mul!(
    x::AbstractVector,
    S::CompressorAdjoint{SparseSignRecipe},
    y::AbstractVector,
    alpha::Number,
    beta::Number,
)
    # Check the compatability of the sizes of the things being multiplied
    vec_mul_dimcheck(x, S, y)
    mul!(x, S.parent.op', y, alpha, beta)
    return nothing
end

# Implement the matrix-Matrix Multiplication operators
# Begin with the left version
function mul!(
    C::AbstractMatrix, S::SparseSignRecipe, A::AbstractMatrix, alpha::Number, beta::Number
)
    left_mat_mul_dimcheck(C, S, A)
    return mul!(C, S.op, A, alpha, beta)
end

# Now implement the right versions
function mul!(
    C::AbstractMatrix, A::AbstractMatrix, S::SparseSignRecipe, alpha::Number, beta::Number
)
    right_mat_mul_dimcheck(C, A, S)
    mul!(C, A, S.op, alpha, beta)
    return nothing
end
