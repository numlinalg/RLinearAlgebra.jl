"""
    Gaussian <: Compressor
An implementation of the Gaussian compression method. This method forms a Gaussian sketch matrix 
    with number of rows or number of columns setting as the compression dimension. 
    
# Mathematical Description
    
Let ``A`` be an ``m \\times n`` matrix that we want to compress. 

If we want to compress ``A`` from the left (i.e., we reduce the number of rows), then 
    we create a Gaussian sketch matrix, ``S``, with dimension ``s \\times m`` where 
    ``s`` is the compression dimension that is supplied by the user. 
    Each entry of ``S`` is generated independently following N(0, 1/s), namely, a 
    Gaussian distribution with mean being zero and variance being 1 divided by the
    compression dimension.
    
If ``A`` is compressed from the right, then we create a Gaussian sketch matrix, ``S``,
    with dimension ``n \\times s``, where ``s`` is the compression dimension that 
    is supplied by the user. 
    Each entry of ``S`` is generated independently following N(0, 1/s), namely, a 
    Gaussian distribution with mean being zero and variance being 1 divided by the 
    compression dimension.

# Fields

- `cardinality::Type{<:Cardinality}`, the direction the compression matrix is intended to be 
    applied to a target matrix or operator. Values allowed are `Left` or `Right`.
- `compression_dim::Int64`, the target compression dimension. Referred to as ``s`` in the
    mathematical description. 

# Constructor

    Gaussian(;cardinality=Left, compression_dim=2)

## Arguments 
- `cardinality::Type{<:Cardinality}`, the direction the compression matrix is intended to be 
    applied to a target matrix or operator. Values allowed are `Left` or `Right`. By default 
    `Left` is chosen. 
- `compression_dim`, the target compression dimension. Referred to as ``s`` in the
    mathemtical description. By default this is set to 2.

## Returns 
- A `Gaussian` object. 
"""

struct Gaussian <: Compressor
    cardinality::Type{<:Cardinality}
    compression_dim::Int64
    # Check on the compression dimension
    Gaussian(cardinality, compression_dim) = begin
        if compression_dim < 0
            throw(ArgumentError("Field 'Compression_dim' must be positive."))
        end

        return new(cardinality, compression_dim)
    end

end

function Gaussian(;cardinality::Type{<:Cardinality} = Left, compression_dim::Int64 = 2)
    # Partially construct the Gaussian datatype
    return Gaussian(cardinality, compression_dim)
end

"""
    GaussianRecipe <: CompressorRecipe
The recipe containing all allocations and information for the Gaussian compressor.
# Fields
- `cardinality::Type{<:Cardinality}`, the direction the compression matrix is intended to be 
    applied to a target matrix or operator. Values allowed are `Left` or `Right`.
- `n_rows::Int64`, the number of rows of the compression matrix.
- `n_cols::Int64`, the number of columns of the compression matrix.
- `scale::Number`, the standard deviation of Gaussian distribution during the compression matrix generation. 
- `op::Matrix{Float64}`, the Gaussian compression matrix.
"""
mutable struct GaussianRecipe <: CompressorRecipe
    cardinality::Type{<:Cardinality}
    compression_dim::Int64
    n_rows::Int64
    n_cols::Int64
    scale::Float64
    op::Matrix{Float64} 
end

function complete_compressor(ingredients::Gaussian, A::AbstractMatrix, type::DataType=eltype(A))
    if ingredients.cardinality == Left
        n_rows = ingredients.compression_dim
        n_cols = size(A, 1)
        initial_size = n_cols
    else
        n_rows = size(A, 2)
        n_cols = ingredients.compression_dim
        initial_size = n_rows
    end

    # Generate entry values by N(0,1/d)
    T = type
    scale = convert(T, 1 / sqrt(ingredients.compression_dim))
    op = scale .* randn(T, n_rows, n_cols)
    return GaussianRecipe(ingredients.cardinality, ingredients.compression_dim, n_rows, n_cols, scale, op)
end

# Allocations in this function are entirely due to bitrand call
function update_compressor!(S::GaussianRecipe)
    # Inplace update of sketch_matrix
    randn!(S.op)
    lmul!(S.scale, S.op)
    return
end

# Implement the matrix-vector multiplication
# Do the right version
function mul!(x::AbstractVector, S::GaussianRecipe, y::AbstractVector, alpha::Number, beta::Number)
    # Check the compatability of the sizes of the things being multiplied
    vec_mul_dimcheck(x, S, y)
    mul!(x, S.op, y, alpha, beta)
    return
end

function mul!(x::AbstractVector, S::CompressorAdjoint{GaussianRecipe}, y::AbstractVector, alpha::Number, beta::Number)
    # Check the compatability of the sizes of the things being multiplied
    vec_mul_dimcheck(x, S, y)
    mul!(x, S.parent.op', y, alpha, beta)
    return
end

# Implement the matrix-Matrix Multiplication operators
# Begin with the left version
function mul!(C::AbstractMatrix, S::GaussianRecipe, A::AbstractMatrix, alpha::Number, beta::Number)
    left_mat_mul_dimcheck(C, S, A) 
    # Built-in multiplication
    mul!(C, S.op, A, alpha, beta)
    return
end

# Now implement the right versions
function mul!(C::AbstractMatrix, A::AbstractMatrix, S::GaussianRecipe, alpha::Number, beta::Number)
    right_mat_mul_dimcheck(C, A, S)
    # Built-in multiplication
    mul!(C, A, S.op, alpha, beta)
    return
end