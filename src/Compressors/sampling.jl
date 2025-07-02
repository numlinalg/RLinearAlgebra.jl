"""
    Sampling <: Compressor

An implementation of the sampling compression method. This method selected 
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

    Sampling(;cardinality = Left(), compression_dim = 8, distribution, type = Float64)

## Arguments

  - `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
    By default `Left()` is chosen.
  - `compression_dim::Int64`, the target compression dimension. Referred to as ``s`` in the
    mathemtical description. By default this is set to 2.
  - `distribution::Function, the function that returns a probability vector over indices.

## Returns

  - A `Sampling` object.

## Throws

  - `ArgumentError` if `compression_dim` is non-positive
"""
struct Sampling <: Compressor
    cardinality::Cardinality
    compression_dim::Int64
    distribution::Distribution # Function that returns a probability vector over indices
    function Sampling(cardinality, compression_dim, distribution)
        # the compression dimension must be positive
        if compression_dim <= 0
            throw(ArgumentError("Field `compression_dim` must be positive."))
        end

        if cardinality == Undef()
            throw(ArgumentError("`cardinality` must be specified as `Left()` or `Right()`.\
            `Undef()` is not allowed in `CountSketch` structure."))
        end

        return new(cardinality, compression_dim, distribution)
    end
end

function Sampling(;
    cardinality::Cardinality=Left(),
    compression_dim::Int64=2,
    distribution::Distribution=Uniform()
)
    # Partially construct the Sampling datatype
    return Sampling(cardinality, compression_dim, distribution)
end

"""
    SamplingRecipe{C<:Cardinality} <: CompressorRecipe

The recipe containing all allocations and information for the sampling compressor. 
  Specify cardinality is for the different implementations of `mul!`.

# Fields

  - `cardinality::Cardinality`, the cardinality of the compressor. The
    value is either `Left()` or `Right()`.
  - `compression_dim::Int64`, the target compression dimension.
  - `n_rows::Int64`, number of rows of compression matrix.
  - `n_cols::Int64`, number of columns of compression matrix.
  - `distribution::Distribution`
  - `idx::Vector{Int64}`, the index set that contains all the chosen indices.
  - `idx_v::SubArray`, the view of the `idx`.
"""
mutable struct SamplingRecipe{C<:Cardinality} <: CompressorRecipe
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

function complete_compressor(sub_sampling::Sampling, A::AbstractMatrix)
    n_rows, n_cols, _ = get_dims(sub_sampling.compression_dim, sub_sampling.cardinality, A)
    # Pull out the variables from ingredients
    compression_dim = sub_sampling.compression_dim
    sub_sampling.distribution.cardinality = sub_sampling.cardinality
    # Compute the weight for each index
    dist_recipe = complete_distribution(sub_sampling.distribution, A)
    idx = Vector{Int64}(undef, compression_dim)
    idx_v = view(idx,:)
    # Randomly generate samples from index set based on weights
    sample_distribution!(idx, dist_recipe)
    return SamplingRecipe{typeof(sub_sampling.cardinality)}(sub_sampling.cardinality, 
                                                            compression_dim, 
                                                            n_rows, 
                                                            n_cols, 
                                                            dist_recipe, 
                                                            idx, 
                                                            idx_v)
end

function update_compressor!(S::SamplingRecipe, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    update_distribution!(S.distribution_recipe, x, A, b)
    # Randomly generate samples from index set based on weights
    sample_distribution!(S.idx_v, S.distribution_recipe)
end
    
# Matrix-matrix multiplication
# Begin with the left version
function mul!(
    C::AbstractArray, 
    S::SamplingRecipe{Left}, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    # Check the compatability of the sizes of the things being multiplied
    left_mul_dimcheck(C, S, A)
    for i in 1:length(S.idx_v)
        c_row = view(C, i, :)
        a_row = view(A, S.idx[i], :)

        c_row .= beta .* c_row .+ alpha .* a_row
    end

    return nothing
end

function mul!(
    C::AbstractArray, 
    S::SamplingRecipe{Right}, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    # Check the compatability of the sizes of the things being multiplied
    left_mul_dimcheck(C, S, A)
    C .*= beta

    for i in 1:length(S.idx_v)
        c_row = view(C, S.idx[i], :)
        a_row = view(A, i, :)

        c_row .+= alpha .* a_row
    end

    return nothing
end

# right version
function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    S::SamplingRecipe{Right}, 
    alpha::Number, 
    beta::Number
)
    # Check the compatability of the sizes of the things being multiplied
    right_mul_dimcheck(C, A, S)
    for i in 1:length(S.idx_v)
        c_col = view(C, :, i)
        a_col = view(A, :, S.idx[i])
        
        c_col .= beta .* c_col .+ alpha .* a_col
    end
    
    return nothing
end

function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    S::SamplingRecipe{Left}, 
    alpha::Number, 
    beta::Number
)
    # Check the compatability of the sizes of the things being multiplied
    right_mul_dimcheck(C, A, S)
    C .*= beta

    for i in 1:length(S.idx_v)
        c_col = view(C, :, S.idx[i])
        a_col = view(A, :, i)
        
        c_col .+= alpha .* a_col
    end
    
    return nothing
end