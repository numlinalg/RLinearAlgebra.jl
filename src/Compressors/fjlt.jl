"""
    FJLT <: Compressor

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
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
- `compression_dim::Int64`, the target compression dimension. Referred to as ``s`` in the
    mathematical description.
- `nnz::Int64`, the target number of nonzeros for each column or row of the spares sign
    matrix.
- `type::Type{<:Number}`, the type of the elements in the compressor.

# Constructor

    SparseSign(;carinality=Left(), compression_dim=2, nnz::Int64=8, type=Float64)

## Keywords
- `carinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
    By default `Left()` is chosen.
- `compression_dim`, the target compression dimension. Referred to as ``s`` in the
    mathemtical description. By default this is set to 2.
- `nnz::Int64`, the number of non-zeros per row/column in the sampling matrix. By default
    this is set to min(compressiond_dim, 8).
- `type::Type{<:Number}`, the type of elements in the compressor.

## Returns
- A `SparseSign` object.

## Throws
- `ArgumentError` if `compression_dim` is non-positive, if `nnz` is exceeds
    `compression_dim`, or if `nnz` is non-positive.
"""
struct FJLT <: Compressor
    cardinality::Cardinality
    compression_dim::Int64
    type::Type{<:Number}
    # perform checks on the number of non-zeros
    function SparseSign(cardinality, compression_dim, nnz, type)
        # the compression dimension must be positive and larger than the number of 
        # nonzeros
        if compression_dim <= 0
            throw(ArgumentError("Field `compression_dim` must be positive."))
        elseif nnz > compression_dim
            throw(
                ArgumentError("Number of non-zero indices, $nnz, must be less than \
                or equal to compression dimension, $compression_dim."
                )
            )
        end

        return new(cardinality, compression_dim, nnz, type)
    end
end

function FJLT(;
    cardinality=Left(),
    compression_dim::Int64=2,
    type::Type{N}=Float64,
) where {N<:Number}
    # Partially construct the sparse sign datatype
    return SparseSign(cardinality, compression_dim, type)
end

"""
    FJLT{C<:Cardinality} <: CompressorRecipe

The recipe containing all allocations and information for the SparseSign compressor.

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to
be applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
- `n_rows::Int64`, the number of rows of the compression matrix.
- `n_cols::Int64`, the number of columns of the compression matrix.
- `scale::Vector{Number}`, the set of values of the non-zero entries of the Spares Sign
compression matrix.
- `op::SparseMatrixCSC`, the Spares Sign compression matrix.
- `signs::Vector{Bool}`, the vector of signs.

# Constructors

    SparseSignRecipe(
        cardinality::C where C<:Cardinality,
        compression_dim::Int64, 
        A::AbstractMatrix, 
        nnz::Int64, 
        type::Type{<:Number}
    )

An external constructor of `SparseSignRecipe` that is dispatched based on the 
value of `cardinality`. See [SparseSign](@ref) for additional details. 

## Arguments 
- `cardinality::C where C<:Cardinality`, the cardinality of the compressor. The 
    value is either `Left()` or `Right()`
- `compression_dim::Int64`, the target compression dimension.
- `A::AbstractMatrix`, a target matrix for compression. 
- `nnz::Int64`, the number of nonzeros in the Sparse Sign compression matrix.
- `type::Type{<:Number}`, the data type for the entries of the compression matrix.

## Returns 
- A `SparseSignRecipe` object.

!!! warning "Use `complete_compressor`"
    While an external constructor is provided, it is mainly for internal use.
    To ensure cross library compatibility please use [`complete_compressor`](@ref)
    for forming the `SparseSignRecipe`.
"""
mutable struct FJLT{C<:Cardinality} <: CompressorRecipe
    cardinality::C
    n_rows::Int64
    n_cols::Int64
    nnz::Int64
    scale::Vector{<:Number}
    op::Union{SparseMatrixCSC,Adjoint{T,SparseMatrixCSC{T,I}}} where {T<:Number,I<:Integer}
    sign::Vector{Bool}
    padding::AbstractMatrix
end


function SparseSignRecipe(
    cardinality::Left,
    compression_dim::Int64,
    A::AbstractMatrix,
    nnz::Int64,
    type::Type{<:Number},
)
    # For compressing from the left, the compressor's dimensions should be 
    # compression_dim by size(A, 1)
    n_rows = compression_dim
    n_cols = size(A, 1)
    initial_dim = n_cols

    # Assign non-zeros of sparse sign matrix 
    total_nnz = initial_dim * nnz
    idxs = Vector{Int64}(undef, total_nnz)
    sparse_idx_update!(idxs, compression_dim, initial_dim, nnz)

    # store the number in the type equivalent to the matrix A
    sc = convert(type, 1 / sqrt(nnz))

    # store as a vector to prevent reallocation during update
    scale = [-sc, sc]
    signs = rand(scale, total_nnz)

    # Allocate the column pointers
    col_ptr = collect(1:nnz:(total_nnz + 1))
    op = SparseMatrixCSC{type,Int64}(n_rows, n_cols, col_ptr, idxs, signs)

    return SparseSignRecipe{typeof(cardinality)}(
        cardinality, n_rows, n_cols, nnz, scale, op
    )
end

function SparseSignRecipe(
    cardinality::Right,
    compression_dim::Int64,
    A::AbstractMatrix,
    nnz::Int64,
    type::Type{<:Number},
)
    # For compressing from the right, the dimension of the compression matrix 
    # should be size(A, 2) by compression_dim 
    n_rows = size(A, 2)
    n_cols = compression_dim
    initial_dim = n_rows

    # Assign non-zeros of compression matrix 
    total_nnz = initial_dim * nnz
    idxs = Vector{Int64}(undef, total_nnz)
    sparse_idx_update!(idxs, compression_dim, initial_dim, nnz)

    # store the number in the type equivalent to the matrix A
    sc = convert(type, 1 / sqrt(nnz))

    # store as a vector to prevent reallocation during update
    scale = [-sc, sc]
    signs = rand(scale, total_nnz)

    # Allocate the column pointers from 1:total_nnz+1
    col_ptr = collect(1:nnz:(total_nnz + 1))
    op = adjoint(SparseMatrixCSC{type,Int64}(n_cols, n_rows, col_ptr, idxs, signs))

    return SparseSignRecipe{typeof(cardinality)}(
        cardinality, n_rows, n_cols, nnz, scale, op
    )
end

function complete_compressor(ingredients::FJLT, A::AbstractMatrix)
        padded_size = Int64(2^(div(log(2, n), 1) + 1)) 
        # Find nearest power 2 and allocate
        padded_matrix = zeros(m, padded_size)
        # Pad matrix and constant vector
        Av = view(A, :, 1:n)
        @views type.Ap[:, 1:n] .= A
    return SparseSignRecipe(
        ingredients.cardinality,
        ingredients.compression_dim,
        A,
        ingredients.nnz,
        ingredients.type,
    )
end

function update_compressor!(S::FJLT)
    (compression_dim, initial_dim) = size(S)
    
    # Generate a new sparse matrix
    S.op = sprand(S.op.n_rows, S.op.n_cols, S.nnz)
    # Resample the non-zero values 
    rand!(S.signs)

    return nothing
end

# Calculates S.op * A and stores it in C 
function mul!(
    C::AbstractArray, 
    S::SparseSignRecipe, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    left_mul_dimcheck(C, S, A)
    return mul!(C, S.op, A, alpha, beta)
end

# Calculates A * S.op and stores it in C 
function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    S::SparseSignRecipe, 
    alpha::Number, 
    beta::Number
)
    right_mul_dimcheck(C, A, S)
    mul!(C, A, S.op, alpha, beta)
    return nothing
end
