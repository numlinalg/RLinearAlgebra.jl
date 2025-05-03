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
- `compression_dim::Int64`, the target compression dimension. Referred to as ``s`` in the
    mathematical description. By default this is set to 2.
- `block_size::Int64`, the number of vectors in the padding matrix.
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
    block_size::Int64
    sparsity::Float64
    type::Type{<:Number}
    # perform checks on the number of non-zeros
    function FJLT(cardinality, compression_dim, block_size, sparsity, type)
        # the compression dimension must be positive and larger than the number of 
        # nonzeros
        if compression_dim <= 0
            throw(ArgumentError("Field `compression_dim` must be positive."))
        elseif sparsity > 1
            throw(ArgumentError("Field `sparsity` must be less than 1."))
        elseif block_size <= 0
            throw(ArgumentError("Field `block_size` must be positive."))
        end

        return new(cardinality, compression_dim, block_size, sparsity, type)
    end
end

function FJLT(;
    cardinality=Left(),
    compression_dim::Int64=2,
    block_size::Int64=10,
    sparsity::Float64=0.0,
    type::Type{N}=Float64,
) where {N<:Number}
    # Partially construct the sparse sign datatype
    return FJLT(cardinality, compression_dim, block_size, sparsity, type)
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


function FJLTRecipe(
    compression_dim::Int64, 
    block_size::Int64,
    cardinality::Left,
    sparsity::Float64,
    A::AbstractMatrix, 
    type::Type{<:Number}
)
    # For compressing from the left, the compressor's dimensions should be 
    # compression_dim by smallest power of 2 larger than size(A, 1)
    n_rows = compression_dim
    a_rows = size(A, 1)
    # Find nearest power 2 and allocate
    padded_size = Int64(2^(ceil(log(2, a_rows))))
    n_cols = size(A, 2) 
    # Generate the padded matrix and signs which need padded size
    padded_matrix = zeros(type, padded_size, block_size)
    signs = bitrand(padded_size)
    sparsity = sparsity == 0.0 ? .25 * log(size(A,2))^2 / n_rows : sparsity
    # generate sparse matrix nonzero gaussian entries occuring with probability sparsity
    sparse_mat = sprandn(type, n_rows, padded_size, sparsity)
    scaling = 1 / (sqrt(padded_size) *  sqrt(sparsity) * sqrt(n_cols))
    return FJLTRecipe{typeof(cardinality), typeof(sparse_mat), typeof(padded_matrix)}(
        cardinality,
        n_rows,
        n_cols,
        sparsity,
        scaling,
        sparse_mat,
        signs,
        padded_matrix
    )
end

function FJLTRecipe(
    compression_dim::Int64, 
    block_size::Int64,
    cardinality::Right, 
    sparsity::Float64,
    A::AbstractMatrix, 
    type::Type{<:Number}
)
    # For compressing from the right, the compressor's dimensions should be 
    # compression_dim by smallest power of 2 larger than size(A, 2)
    n_cols = compression_dim
    a_cols = size(A, 2)
    # Find nearest power 2 and allocate
    padded_size = Int64(2^(ceil(log(2, a_cols))))
    n_rows = size(A, 2) 
    # Generate the padded matrix and signs which need padded size
    padded_matrix = zeros(type, block_size, padded_size)
    signs = bitrand(padded_size)
    sparsity = sparsity == 0.0 ? .25 * log(size(A,2))^2 / n_rows : sparsity
    # generate sparse matrix nonzero gaussian entries occuring with probability sparsity
    sparse_mat = sprandn(type, padded_size, n_cols, sparsity)
    scaling = 1 / (sqrt(padded_size) *  sqrt(sparsity) * sqrt(n_cols))
    return FJLTRecipe{typeof(cardinality), typeof(sparse_mat), typeof(padded_matrix)}(
        cardinality,
        n_rows,
        n_cols,
        sparsity,
        scaling,
        sparse_mat,
        signs,
        padded_matrix
    )
end

function complete_compressor(ingredients::FJLT, A::AbstractMatrix)
    sparsity = ingredients.sparsity
    # Pad matrix and constant vector
    return FJLTRecipe(
        ingredients.compression_dim, 
        ingredients.block_size,
        ingredients.cardinality,
        ingredients.sparsity,
        A, 
        ingredients.type
    )
end

function update_compressor!(S::FJLTRecipe)
    n_rows, n_cols = size(S.op)
    type = eltype(S.op)
    # Generate a new sparse matrix
    S.op = sprandn(type, n_rows, n_cols, S.sparsity)
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
    c_rows, c_cols = size(C, 1), size(C, 2)
    s_rows, s_cols = size(S, 1), size(S, 2)
    a_rows, a_cols = size(A, 1), size(A, 2)
    type = eltype(S.op)
    b_size = size(S.padding, 2)
    if c_rows != s_rows
        throw(
            DimensionMismatch("Matrix C has $c_rows rows while S has $s_rows rows.")
        )
    elseif c_cols != a_cols
        throw(
            DimensionMismatch("Matrix C has $c_cols columns while A has $a_cols columns.")
        )
    #=elseif a_rows > s_cols 
        throw(
            DimensionMismatch("Matrix A has more rows than the matrix S has columns.")
        )=#
    end

    # To be memory efficient we apply FJLT block-wise in column blocks
    last_block_size = rem(a_cols, b_size)
    # Compute the number of full block iterations that will be needed
    n_iter = div(a_cols, b_size)
    start_col = 1
    for i in 1:n_iter
        # Working with blocks requires views of the matrix
        last_col = start_col + b_size - 1
        Av = view(A, :, start_col:last_col)
        Cv = view(C, :, start_col:last_col)
        pv = view(S.padding, 1:a_rows, :)
        # ensure that padding matrix is set to zeros
        fill!(S.padding, zero(type))
        # Copy the matrix A to the padding matrix
        copyto!(pv, Av)
        # Apply signs and fwht to the padding matrix
        for i in 1:b_size
            pv = view(S.padding, :, i)
            fwht!(pv, S.signs, scaling = S.scale) 
        end

        # Apply the operator to the matrix
        mul!(Cv, S.op, S.padding, alpha, beta)
        start_col = last_col + 1
    end
    
    # Handle the last block that is last than the block size
    if last_block_size > 0
        last_col = start_col + last_block_size - 1
        Av = view(A, :, start_col:last_col)
        Cv = view(C, :, start_col:last_col)
        pv = view(S.padding, 1:a_rows, 1:last_block_size)
        # ensure that padding matrix is set to zeros
        fill!(S.padding, zero(type))
        # Copy the matrix A to the padding matrix
        copyto!(pv, Av)
        # Apply signs and fwht to the padding matrix
        for i in 1:b_size
            pv = view(S.padding, :, i)
            fwht!(pv, S.signs, scaling = S.scale) 
        end
        
        # Perform accesses only up to the entries
        pv = view(S.padding, :, 1:last_block_size)
        # Apply the operator to the matrix
        mul!(Cv, S.op, pv, alpha, beta)
    end

    return nothing 
end

# Calculates S.op * A and stores it in C  when S has left cardinality
function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    S::FJLTRecipe{Left, <:SparseMatrixCSC, <:AbstractMatrix}, 
    alpha::Number, 
    beta::Number
)
    # Here padding does not allow us to use standard dim check
    # instead we check that rows of C == rows of S and cols of C == cols of A, but we only 
    # cheeck that the cols of S > rows of A
    c_rows, c_cols = size(C, 1), size(C, 2)
    s_rows, s_cols = size(S, 1), size(S, 2)
    a_rows, a_cols = size(A, 1), size(A, 2)
    b_size = size(S.padding, 2)
    type = eltype(S.op)
    if c_rows != a_rows
        throw(
            DimensionMismatch("Matrix C has $c_rows rows while A has $a_rows rows.")
        )
    elseif c_cols > s_cols 
        throw(
            DimensionMismatch("Matrix C has more columns and S.")
        )
    elseif a_cols != s_rows
        throw(
            DimensionMismatch("Matrix A has $a_cols columns while S has $s_rows rows.")
        )
    end

    # To be memory efficient we apply FJLT block-wise in column blocks
    last_block_size = rem(a_rows, b_size)
    # Compute the number of full block iterations that will be needed
    n_iter = div(c_rows, b_size)
    start_row = 1
    for i in 1:n_iter
        # Working with blocks requires views of the matrix
        last_row = start_row + b_size - 1
        Av = view(A, start_row:last_row, :)
        Cv = view(C, start_row:last_row, 1:c_cols)
        pv = view(S.padding, :, :)
        # Everything should be stored in the transpose of padding matrix because of 
        # left padding matrix is strucuted with more rows than columns
        mul!(pv', Av, S.op, alpha, zero(type))
        # Apply signs and fwht to the padding matrix
        for i in 1:b_size
            pv = view(S.padding, :, i)
            fwht!(pv, S.signs, scaling = S.scale) 
        end
       
        pv = view(S.padding, 1:c_cols, :)
        # add the result to C note that because of padding instead of returning padded 
        # matrix we only return the part that corresponds to the dimensions of C
        axpby!(one(type), pv', beta, Cv)
        start_row = last_row + 1
    end
    
    # Handle the last block that is last than the block size
    if last_block_size > 0
        last_row = start_row + last_block_size - 1
        Av = view(A, start_row:last_row, :)
        Cv = view(C, start_row:last_row, 1:c_cols)
        pv = view(S.padding, :, 1:last_block_size)
        # Everything should be stored in the transpose of padding matrix because of 
        # left padding matrix is strucuted with more rows than columns
        mul!(pv', Av, S.op, alpha, zero(type))
        # Apply signs and fwht to the padding matrix
        for i in 1:last_block_size
            pv = view(S.padding, :, i)
            fwht!(pv, S.signs, scaling = S.scale) 
        end
        
        pv = view(S.padding, 1:c_cols, 1:last_block_size)
        # add the result to C note that because of padding instead of returning padded 
        # matrix we only return the part that corresponds to the dimensions of C
        axpby!(one(type), pv', beta, Cv)
    end

    return nothing 
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
    c_rows, c_cols = size(C, 1), size(C, 2)
    s_rows, s_cols = size(S, 1), size(S, 2)
    a_rows, a_cols = size(A, 1), size(A, 2)
    type = eltype(S.op)
    b_size = size(S.padding, 1)
    if c_rows != a_rows
        throw(
            DimensionMismatch("Matrix C has $c_rows rows while A has $a_rows rows.")
        )
    elseif c_cols > s_cols 
        throw(
            DimensionMismatch("Matrix C has more columns and S.")
        )
    elseif a_cols > s_rows
        throw(
            DimensionMismatch("Matrix A has $a_cols columns while S has $s_rows rows.")
        )
    end

    # To be memory efficient we apply FJLT block-wise in column blocks
    last_block_size = rem(a_rows, b_size)
    # Compute the number of full block iterations that will be needed
    n_iter = div(a_rows, b_size)
    start_row = 1
    for i in 1:n_iter
        # Working with blocks requires views of the matrix
        last_row = start_row + b_size - 1
        Av = view(A, start_row:last_row, :)
        Cv = view(C, start_row:last_row, :)
        pv = view(S.padding, :, 1:a_cols)
        # ensure that padding matrix is set to zeros
        fill!(S.padding, zero(type))
        # Copy the matrix A to the padding matrix
        copyto!(pv, Av)
        # Apply signs and fwht to the padding matrix
        for i in 1:b_size
            pv = view(S.padding, i, :)
            fwht!(pv, S.signs, scaling = S.scale) 
        end

        # Apply the operator to the matrix
        mul!(Cv, S.padding, S.op, alpha, beta)
        start_row = last_row + 1
    end
    
    # Handle the last block that is last than the block size
    if last_block_size > 0
        last_row = start_row + last_block_size - 1
        Av = view(A, start_row:last_row, :)
        Cv = view(C, start_row:last_row, :)
        pv = view(S.padding, 1:last_block_size, 1:a_cols)
        # ensure that padding matrix is set to zeros
        fill!(S.padding, zero(type))
        # Copy the matrix A to the padding matrix
        copyto!(pv, Av)
        # Apply signs and fwht to the padding matrix
        for i in 1:last_block_size
            pv = view(S.padding, i, :)
            fwht!(pv, S.signs, scaling = S.scale) 
        end

        pv = view(S.padding, 1:last_block_size, :)
        # Apply the operator to the matrix
        mul!(Cv, pv, S.op, alpha, beta)
    end

    return nothing
end

function mul!(
    C::AbstractArray, 
    S::FJLTRecipe{Right, <:SparseMatrixCSC, <:AbstractMatrix}, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    # Here padding does not allow us to use standard dim check
    # instead we check that rows of C == rows of A and cols of C == cols of S, but we only 
    # cheeck that the rows of S > cols of A
    c_rows, c_cols = size(C, 1), size(C, 2)
    s_rows, s_cols = size(S, 1), size(S, 2)
    a_rows, a_cols = size(A, 1), size(A, 2)
    b_size = size(S.padding, 1)
    type = eltype(S.op)
    if c_cols != a_cols
        throw(
            DimensionMismatch("Matrix C has $c_cols cols while A has $a_cols cols.")
        )
    elseif c_rows > s_rows 
        throw(
            DimensionMismatch("Matrix C has more rows than S.")
        )
    elseif a_rows > s_cols
        throw(
            DimensionMismatch("Matrix A has $a_rows rows while S has $s_cols cols.")
        )
    end

    # To be memory efficient we apply FJLT block-wise in column blocks
    last_block_size = rem(a_cols, b_size)
    # Compute the number of full block iterations that will be needed
    n_iter = div(a_cols, b_size)
    start_col = 1
    for i in 1:n_iter
        # Working with blocks requires views of the matrix
        last_col = start_col + b_size - 1
        Av = view(A, :, start_col:last_col)
        Cv = view(C, :, start_col:last_col)
        pv = view(S.padding, :, :)
        # Everything should be stored in the transpose of padding matrix because of 
        # left padding matrix is strucuted with more rows than columns
        mul!(pv', S.op, Av, alpha, zero(type))
        # Apply signs and fwht to the padding matrix
        for i in 1:b_size
            pv = view(S.padding, i, :)
            fwht!(pv, S.signs, scaling = S.scale) 
        end
       
        pv = view(S.padding, :, 1:c_rows)
        # add the result to C note that because of padding instead of returning padded 
        # matrix we only return the part that corresponds to the dimensions of C
        axpby!(one(type), pv', beta, Cv)
        start_col = last_col + 1
    end
    
    # Handle the last block that is last than the block size
    if last_block_size > 0
        last_col = start_col + last_block_size - 1
        Av = view(A, :, start_col:last_col)
        Cv = view(C, :, start_col:last_col)
        pv = view(S.padding, 1:last_block_size, :)
        # Everything should be stored in the transpose of padding matrix because of 
        # left padding matrix is strucuted with more rows than columns
        mul!(pv', S.op, Av,  alpha, zero(type))
        # Apply signs and fwht to the padding matrix
        for i in 1:last_block_size
            pv = view(S.padding, i, :)
            fwht!(pv, S.signs, scaling = S.scale) 
        end
       
        pv = view(S.padding, 1:last_block_size, 1:c_rows)
        # add the result to C note that because of padding instead of returning padded 
        # matrix we only return the part that corresponds to the dimensions of C
        axpby!(one(type), pv', beta, Cv)
    end

    return nothing
end
