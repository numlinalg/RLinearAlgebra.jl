"""
    FJLT <: Compressor

An implementation of the Fast Johnson-Lindesntrauss Tranform method. This technique applies
sparse matrix, a Walsh-Hadamard transform, and diagonal sign matrix to produce a sketch. See
[ailon2009fast](@cite) for additional details.

# Mathematical Description

Let ``A`` be an ``m \\times n`` matrix that we want to compress. If we want
to compress ``A`` from the left (i.e., we reduce the number of rows), then
we create a matrix, ``S``, with dimension ``s \\times m`` where
``s`` is the compression dimension that is supplied by the user.
Here ``S=KHD`` where ``K`` is a sparse matrix with  with dimension ``s \\times m``, ``H`` is
a Hadamard matrix of dimension ``m \\times m``, ``D`` of is a diagonal matrix with random
``\\pm 1`` on the diagonal of dimensions ``m \\times m``. 
We then sketch ``A`` by applying ``SHD`` from the left. ``S`` is a sparse
matrix with with sparsity ``1-q``, and the nonzero entries being independent draws from a 
normal distribution with mean ``0`` and variance ``1/q``. We do not form the matrix ``H`` 
instead we apply the Fast Walsh-Hadamard transform to the matrix ``DA``, which has a 
``m\\log(m)`` computational complexity instead of ``m^2``. 

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
- `compression_dim::Int64`, the target compression dimension. Referred to as ``s`` in the
    mathematical description.
- `sparsity::Int64`, the desired sparsity of the matrix ``K``, by default sparsity will be 
    set to be ``\\min\\left(1/4 \\log(n)^2 / m, 1\\right)``, see [ailon2009fast](@cite).
- `type::Type{<:Number}`, the type of the elements in the compressor.

# Constructor

    FJLT(;carinality=Left(), compression_dim=2, sparsity=0.0, type=Float64)

## Keywords
- `carinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
    By default `Left()` is chosen.
- `compression_dim`, the target compression dimension. Referred to as ``s`` in the
    mathemtical description. By default this is set to 2.
- `sparsity::Int64`, the desired sparsity of the matrix ``K``, by default sparsity will be 
    set to be ``\\min\\left(1/4 \\log(n)^2 / m, 1\\right)``, see [ailon2009fast](@cite).
- `type::Type{<:Number}`, the type of elements in the compressor.

## Returns
- A `FJLT` object.

## Throws
- `ArgumentError` if `compression_dim` is non-positive, if `nnz` is exceeds
    `compression_dim`, or if `sparisty` is greater than 1.
"""
struct FJLT <: Compressor
    cardinality::Cardinality
    compression_dim::Int64
    sparsity::Float64
    type::Type{<:Number}
    # perform checks on the number of non-zeros
    function FJLT(cardinality, compression_dim, sparsity, type)
        # the compression dimension must be positive and larger than the number of 
        # nonzeros
        if compression_dim <= 0
            throw(ArgumentError("Field `compression_dim` must be positive."))
        elseif sparsity > 1
            throw(ArgumentError("Field `sparsity` must be less than 1."))
        end

        return new(cardinality, compression_dim, sparsity, type)
    end
end

function FJLT(;
    cardinality=Left(),
    compression_dim::Int64=2,
    sparsity::Float64=0.0,
    type::Type{N}=Float64,
) where {N<:Number}
    # Partially construct the sparse sign datatype
    return FJLT(cardinality, compression_dim, sparsity, type)
end

"""
    FJLTRecipe <: CompressorRecipe

The recipe containing all allocations and information for the SparseSign compressor.

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to
be applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
- `n_rows::Int64`, the number of rows of the compression matrix.
- `n_cols::Int64`, the number of columns of the compression matrix.
- `sparsity::Vector{Number}`, the expected sparsity of the Sparse operator matrix.
- `op::SparseMatrixCSC`, the Spares Sign compression matrix.
- `signs::BitVector`, the vector of signs.
- `padding::AbstractMatr`, the matrix containing the padding for the matrix being sketched.
"""
mutable struct FJLTRecipe{
    C<:Cardinality, 
    S<:SparseMatrixCSC, 
    M<:AbstractMatrix
   } <: CompressorRecipe
    cardinality::C
    n_rows::Int64
    n_cols::Int64
    sparsity::Float64
    scale::Float64
    op::S
    signs::BitVector
    padding::M
end


function compute_padding(
    compression_dim::Int64, 
    cardinality::Left,
    A::AbstractMatrix, 
    type::Type{<:Number}
)
    # For compressing from the left, the compressor's dimensions should be 
    # compression_dim by smallest power of 2 larger than size(A, 1)
    n_rows = compression_dim
    a_rows = size(A, 1)
    # Find nearest power 2 and allocate
    padded_size = Int64(2^(ceil(log(2, a_rows))))
    n_cols = padded_size 
    # Generate the padded matrix and signs which need padded size
    padded_matrix = zeros(type, padded_size, size(A,2))
    signs = bitrand(padded_size)
    # return the computed dimensions and the unused A dimension for padded matrix construct
    return n_rows, n_cols, size(A, 2), padded_matrix, signs, type(1/sqrt(padded_size))
end

function compute_padding(
    compression_dim::Int64, 
    cardinality::Right, 
    A::AbstractMatrix, 
    type::Type{<:Number}
)
    # For compressing from the right, the compressor's dimensions should be 
    # compression_dim by smallest power of 2 larger than size(A, 2)
    n_cols = compression_dim
    a_cols = size(A, 2)
    # Find nearest power 2 and allocate
    padded_size = Int64(2^(ceil(log(2, a_cols))))
    n_rows = padded_size 
    # Generate the padded matrix and signs which need padded size
    padded_matrix = zeros(type, size(A, 1), padded_size)
    signs = bitrand(padded_size)
    # return the computed dimensions and the unused A dimension for padded matrix construct
    return n_rows, n_cols, size(A, 1), padded_matrix, signs, type(1/sqrt(padded_size))
end

function complete_compressor(ingredients::FJLT, A::AbstractMatrix)
    sparsity = ingredients.sparsity
    # Pad matrix and constant vector
    pad = compute_padding(
        ingredients.compression_dim, 
        ingredients.cardinality, 
        A, 
        ingredients.type
    )
    n_rows = pad[1]
    n_cols = pad[2]
    unused_dim = pad[3]
    pad_mat = pad[4]
    signs = pad[5]
    scaling = pad[6]
    # set default sparsity parameter if not specified
    sparsity = ingredients.sparsity == 0.0 ? .25 * log(unused_dim)^2 / n_rows : sparsity
    # generate sparse matrix nonzero gaussian entries occuring with probability sparsity
    sparse_mat = sprandn(ingredients.type, n_rows, n_cols, sparsity) ./ sqrt(sparsity)
    return FJLTRecipe{typeof(ingredients.cardinality), typeof(sparse_mat), typeof(pad_mat)}(
        ingredients.cardinality,
        n_rows,
        n_cols,
        sparsity,
        scaling,
        sparse_mat,
        signs,
        pad_mat
    )
end

function update_compressor!(S::FJLTRecipe)
    n_rows, n_cols = size(S.op)
    type = eltype(S.op)
    # Generate a new sparse matrix
    S.op = sprandn(type, n_rows, n_cols, S.sparsity) ./ type(sqrt(S.sparsity))
    # Resample the non-zero values 
    rand!(S.signs)

    return nothing
end

# Calculates S.op * A and stores it in C 
function mul!(
    C::AbstractArray, 
    S::FJLTRecipe{Left, <:SparseMatrixCSC, <:AbstractMatrix}, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    # Here padding does not allow us to use standard dim check
    # instead we check that rows of C == rows of S and cols of C == cols of A, but we only 
    # cheeck that the cols of S > rows of A
    c_rows, c_cols = size(C)
    s_rows, s_cols = size(S)
    a_rows, a_cols = size(A)
    if c_rows != s_rows
        throw(
            DimensionMismatch("Matrix C has $c_rows rows while S has $s_rows rows.")
        )
    elseif c_cols != a_cols
        throw(
            DimensionMismatch("Matrix C has $c_cols columns while A has $a_cols columns.")
        )
    elseif a_rows > s_cols
        throw(
            DimensionMismatch("Matrix A has more rows than the matrix S has columns.")
        )
    elseif size(S.padding, 2) < a_cols
        throw(
            DimensionMismatch("Matrix A has more columns than the padding matrix in S.")
        )
    end

    # ensure that padding matrix is set to zeros
    fill!(S.padding, zero(eltype(S.op)))
    # Copy the matrix A to the padding matrix
    pv = view(S.padding, 1:a_rows, :)
    copyto!(pv, A)
    # Apply signs and fwht to the padding matrix
    # Perform this blockwise to accomdate when the number rows does not equal that in the 
    # original matrix
    for i in 1:a_cols
        pv = view(S.padding, :, i)
        fwht!(pv, S.signs, scaling = S.scale) 
    end

    return mul!(C, S.op, S.padding, alpha, beta)
end

# Calculates S.op * A and stores it in C  when S has left cardinality
function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    S::Adjoint{FJLTRecipe{Left, <:SparseMatrixCSC, <:AbstractMatrix}}, 
    alpha::Number, 
    beta::Number
)
    # Here padding does not allow us to use standard dim check
    # instead we check that rows of C == rows of S and cols of C == cols of A, but we only 
    # cheeck that the cols of S > rows of A
    c_rows, c_cols = size(C)
    s_rows, s_cols = size(S)
    a_rows, a_cols = size(A)
    if c_rows != s_rows
        throw(
            DimensionMismatch("Matrix C has $c_rows rows while S has $s_rows rows.")
        )
    elseif c_cols != a_cols
        throw(
            DimensionMismatch("Matrix C has $c_cols columns while A has $a_cols columns.")
        )
    elseif a_rows > s_cols
        throw(
            DimensionMismatch("Matrix A has more rows than the matrix S has columns.")
        )
    elseif size(S.padding, 2) < a_cols
        throw(
            DimensionMismatch("Matrix A has more columns than the padding matrix in S.")
        )
    end
    # With left cardinality first apply the operator matrix to A then place the result in
    # the padding matrix
    mul!(S.padding, S.op, A, alpha, 0.0)
    # Copy the matrix A to the padding matrix
    # Apply signs and fwht to the padding matrix
    for i in 1:a_rows
        pv = view(S.padding, i, :)
        fwht!(pv, S.signs, scaling = S.scale) 
    end

    return axpby!(1.0, beta, C, S.padding)
end

# Calculates A * S.op and stores it in C 
function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    S::FJLTRecipe{Right, <:SparseMatrixCSC, <:AbstractMatrix}, 
    alpha::Number, 
    beta::Number
)
    # Here padding does not allow us to use standard dim check
    # instead we check that rows of C == rows of A and cols of C == cols of S, but we only 
    # cheeck that the rows of S > cols of A
    c_rows, c_cols = size(C)
    s_rows, s_cols = size(S)
    a_rows, a_cols = size(A)
    if c_rows != a_rows
        throw(
            DimensionMismatch("Matrix C has $c_rows rows while A has $a_rows rows.")
        )
    elseif c_cols != s_cols
        throw(
            DimensionMismatch("Matrix C has $c_cols columns while S has $s_cols columns.")
        )
    elseif a_cols > s_rows
        throw(
            DimensionMismatch("Matrix A has more columsn than the matrix S has rows.")
        )
    elseif size(S.padding, 1) < a_rows
        throw(
            DimensionMismatch("Matrix A has more rows than the padding matrix in S.")
        )
    end

    # ensure that padding matrix is set to zeros
    fill!(S.padding, zero(eltype(S.op)))
    # Copy the matrix A to the padding matrix
    pv = view(S.padding, :, 1:a_cols)
    copyto!(pv, A)
    # Apply signs and fwht to the padding matrix
    # Perform this blockwise to accomdate when the number rows does not equal that in the 
    # original matrix
    for i in 1:a_rows
        pv = view(S.padding, i, :)
        fwht!(pv, S.signs, scaling = S.scale) 
    end

    return mul!(C, S.padding, S.op, alpha, beta)
end

function mul!(
    C::AbstractArray, 
    S::Adjoint{FJLTRecipe{Right, <:SparseMatrixCSC, <:AbstractMatrix}}, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    # Here padding does not allow us to use standard dim check
    # instead we check that rows of C == rows of A and cols of C == cols of S, but we only 
    # cheeck that the rows of S > cols of A
    c_rows, c_cols = size(C)
    s_rows, s_cols = size(S)
    a_rows, a_cols = size(A)
    if c_rows != a_rows
        throw(
            DimensionMismatch("Matrix C has $c_rows rows while A has $a_rows rows.")
        )
    elseif c_cols != s_cols
        throw(
            DimensionMismatch("Matrix C has $c_cols columns while S has $s_cols columns.")
        )
    elseif a_cols > s_rows
        throw(
            DimensionMismatch("Matrix A has more columsn than the matrix S has rows.")
        )
    elseif size(S.padding, 1) < a_rows
        throw(
            DimensionMismatch("Matrix A has more rows than the padding matrix in S.")
        )
    end

    mul!(S.padding, S.op, A, alpha, 0.0)
    # Apply signs and fwht to the padding matrix
    for i in 1:a_cols
        pv = view(S.padding, :, i)
        fwht!(pv, S.signs, scaling = S.scale) 
    end

    return axpby!(1.0, beta, C, S.padding)
end
