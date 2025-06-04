"""
    SubCompressor <: Compressor

An implementation of the sub-sampling compression method. This method selected 
the rows/columns of the matrix by given distribution with number of rows/columns 
setting as the compression dimension.

# Mathematical Description

Let ``A`` be an ``m \\times n`` matrix that we want to compress.

If we want to compress ``A`` from the left (i.e., we reduce the number of rows), then
we create an index set to contain all the indices of seletec rows. The indices are 
chosen by sampling over all the rows with given distribution.

If ``A`` is compressed from the right (i.e., we reduce the number of columns), then
we create an index set to contain all the indices of selected columns. The indices 
are chosen by sampling over all the columns with given distribution.

# Fields

  - `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
  - `compression_dim::Int64`, the target compression dimension. Referred to as ``s`` in the
    mathematical description.
  - `distribution::Function, the function that returns a probability vector over indices.
  - `type::Type{<:Number}`, the type of the elements in the compressor.

# Constructor

    SubCompressor(;cardinality = Left(), compression_dim = 8, distribution, type = Float64)

## Arguments

  - `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
    By default `Left()` is chosen.
  - `compression_dim::Int64`, the target compression dimension. Referred to as ``s`` in the
    mathemtical description. By default this is set to 2.
  - `distribution::Function, the function that returns a probability vector over indices.
  - `type::Type{<:Number}`, the type of elements in the compressor.

## Returns

  - A `SubCompressor` object.

## Throws

  - `ArgumentError` if `compression_dim` is non-positive
"""
struct SubCompressor <: Compressor
    cardinality::Cardinality
    compression_dim::Int64
    distribution::Distribution # Function that returns a probability vector over indices
    function SubCompressor(cardinality, compression_dim, distribution)
        # the compression dimension must be positive
        if compression_dim <= 0
            throw(ArgumentError("Field `compression_dim` must be positive."))
        end

        return new(cardinality, compression_dim, distribution)
    end
end

function SubCompressor(;
    cardinality::Cardinality = Left(),
    compression_dim::Int64 = 2,
    distribution::Distribution = Uniform()
)
    # Partially construct the SubCompressor datatype
    return SubCompressor(cardinality, compression_dim, distribution)
end

"""
    SubCompressorRecipe <: CompressorRecipe

The recipe containing all allocations and information for the sub-compressor.

# Fields

  - `cardinality::Cardinality`, the cardinality of the compressor. The
    value is either `Left()` or `Right()`.
  - `state_space`::Vector{Int64}, 
  - `compression_dim::Int64`, the target compression dimension.
  - `distribution::Distribution`
  - `weights::ProbabilityWeights`, the weight vector that indicates the discrete probability of selecting each row/column
  - `idx::Vector{Int64}`, the index set that contains all the chosen indices.
  - `idx_v::SubArray`, the view of the `idx`.
"""
mutable struct SubCompressorRecipe <: CompressorRecipe
    cardinality::Cardinality
    compression_dim::Int64
    n_rows::Int64
    n_cols::Int64
    distribution_recipe::DistributionRecipe
    idx::Vector{Int64}
    idx_v::SubArray
end

function get_dims(compression_dim::Int64, cardinality::Left, A::AbstractMatrix)
    n_rows = compression_dim
    n_cols = size(A, 1)
    initial_size = n_cols
    return n_rows, n_cols, initial_size
end

function get_dims(compression_dim::Int64, cardinality::Right, A::AbstractMatrix)
    n_rows = size(A, 2)
    n_cols = compression_dim
    initial_size = n_rows
    return n_rows, n_cols, initial_size
end

function complete_compressor(subcompressor::SubCompressor, A::AbstractMatrix)
    n_rows, n_cols, _ = get_dims(subcompressor.compression_dim, subcompressor.cardinality, A)
    # Pull out the variables from ingredients
    compression_dim = subcompressor.compression_dim
    subcompressor.distribution.cardinality = subcompressor.cardinality
    # Compute the weight for each index
    dist_recipe = complete_distribution(subcompressor.distribution, A)
    idx = Vector{Int64}(undef, compression_dim)
    idx_v = view(idx,:)
    # Randomly generate samples from index set based on weights
    sample_distribution!(idx_v, idx, dist_recipe)
    return SubCompressorRecipe(subcompressor.cardinality, compression_dim, n_rows, n_cols, dist_recipe, idx, idx_v)
end

function update_compressor!(S::SubCompressorRecipe, A, b, x)
    update_distribution!(S.distribution_recipe, A, b, x)
    # Randomly generate samples from index set based on weights
    sample_distribution!(S.idx_v, S.idx, S.distribution_recipe)
end
    
# Matrix-matrix multiplication
# Begin with the left version
function mul!(
    C::AbstractArray, 
    S::SubCompressorRecipe, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    # Check the compatability of the sizes of the things being multiplied
    left_mul_dimcheck(C, S, A)
    for i in 1:length(S.idx_v)
        C[i,:] .= beta .* C[i,:] + alpha * A[S.idx[i],:]
    end

    return nothing
end

# right version
function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    S::SubCompressorRecipe, 
    alpha::Number, 
    beta::Number
)
    # Check the compatability of the sizes of the things being multiplied
    right_mul_dimcheck(C, A, S)
    for i in 1:length(S.idx_v)
        C[:,i] .= beta .* C[:,i] + alpha * A[:,S.idx[i]]
    end
    
    return nothing
end