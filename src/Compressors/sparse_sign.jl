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
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
- `compression_dim::Int64`, the target compression dimension. Referred to as ``s`` in the
    mathematical description.
- `nnz::Int64`, the target number of nonzeros for each column or row of the spares sign
    matrix.

# Constructor

    SparseSign(;carinality=Left(), compression_dim=2, nnz::Int64=8)

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
            throw(
                ArgumentError("Number of non-zero indices, $nnz, must be less than \
                or equal to compression dimension, $compression_dim."
                )
            )
        elseif nnz <= 0
            throw(ArgumentError("Field `nnz` must be positive."))
        end

        return new(cardinality, compression_dim, nnz, type)
    end
end

function SparseSign(;
    cardinality=Left(),
    compression_dim::Int64=2,
    nnz::Int64=min(8, compression_dim),
    type::Type{N}=Float64,
) where {N<:Number}
    # Partially construct the sparse sign datatype
    return SparseSign(cardinality, compression_dim, nnz, type)
end

"""
    sparse_idx_update!(
        values::Vector{Int64}, 
        max_sample_val::Int64, 
        n_samples::Int64, 
        sample_size::Int64
    )

Implicitly splits `values` into `n_samples` components of size `sample_size`. 
On each component, replaces the entries of each component with a random sample 
without replacement of size `sample_size` from the set `1:max_sample_val`.

!!! warn 
    `values` should have length equal to `sample_size*n_samples`, but this 
    is not checked. 

# Arguments
- `values::Vector{Int64}`, the indice to be replaced.
- `max_sample_val::In64`, implicitly supplies the set from which to sample,
    `1:max_sample_val`.
- `n_samples::Int64`, the components that `values` is implicitly split into. 
- `sample_size::Int64`, the size each component that `values` is split into.

# Returns
- `nothing`
"""
function sparse_idx_update!(
    values::Vector{Int64}, 
    max_sample_val::Int64, 
    n_samples::Int64, 
    sample_size::Int64
)
    first_idx = 1
    for i in 1:n_samples
        # correct for one indexing to find sample size
        last_idx = first_idx + sample_size - 1
        # Sample indices from the intial_size
        idx_view = view(values, first_idx:last_idx)
        sample!(1:max_sample_val, idx_view; replace=false, ordered=true)
        first_idx = last_idx + 1
    end

    return nothing
end

"""
    SparseSignRecipe{C<:Cardinality} <: CompressorRecipe

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
mutable struct SparseSignRecipe{C<:Cardinality} <: CompressorRecipe
    cardinality::C
    n_rows::Int64
    n_cols::Int64
    nnz::Int64
    scale::Vector{<:Number}
    op::Union{SparseMatrixCSC,Adjoint{T,SparseMatrixCSC{T,I}}} where {T<:Number,I<:Integer}
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

function complete_compressor(ingredients::SparseSign, A::AbstractMatrix)
    return SparseSignRecipe(
        ingredients.cardinality,
        ingredients.compression_dim,
        A,
        ingredients.nnz,
        ingredients.type,
    )
end

function update_compressor!(S::SparseSignRecipe{Left})
    (compression_dim, initial_dim) = size(S)
    op = S.op
    nnz = S.nnz
    
    # For each column of `S.op`, choose new non-zero entries at random without 
    # replacement
    sparse_idx_update!(op.rowval, compression_dim, initial_dim, nnz)

    # Resample the non-zero values 
    rand!(op.nzval, S.scale)

    return nothing
end

function update_compressor!(S::SparseSignRecipe{Right})
    (compression_dim, initial_dim) = (S.n_cols, S.n_rows)

    # For each column of `S.op.parent`, choose new non-zero entries at random 
    # withour replacment. Equivalent to: for each row of `S.op.parent`, choose 
    # new non-zero entries at random without replacement.
    op = S.op.parent
    nnz = S.nnz
    sparse_idx_update!(op.rowval, compression_dim, initial_dim, nnz)

    # Update the nonzero values
    rand!(op.nzval, S.scale)

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
