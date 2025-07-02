"""
    SRHT <: Compressor

An implementation of the Subsampled Randomized Hadamard Transform (SRHT) method. This 
technique applies a subsampling matrix, a Walsh-Hadamard transform, and a diagonal sign 
matrix to produce a sketch. See [](@cite) for additional details.

# Mathematical Description

Let ``A`` be an ``m \\times n`` matrix that we want to compress. If we want
to compress ``A`` from the left (i.e., we reduce the number of rows), then
we create a matrix, ``S``, with dimension ``s \\times m`` where
``s`` is the compression dimension that is supplied by the user.
Here ``S=KHD`` where 

- ``K`` is a matrix with  with dimension ``s \\times m``, where the rows are made up of a 
    random sample of the rows of a ``m \\times m`` identity matrix.
- ``H`` is a Hadamard matrix of dimension ``m \\times m``, which is implicitly applied 
    through the fast Walsh-Hadamard transform;
- ``D`` of is a diagonal matrix of dimension ``m \\times m`` with entries given by 
    independent Rademacher variables.

If we want to compress ``A`` from the right (i.e., we reduce the number of columns), then 
we would apply ``S=DHK`` from the right where the dimensions of the matrices are adjusted 
to reflect the number of columns in ``A``.

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
- `compression_dim::Int64`, the target compression dimension. Referred to as ``s`` in the
    mathematical description.
- `block_size::Int64`, the number of vectors in the padding matrix.
- `type::Type{<:Number}`, the type of the elements in the compressor.

# Constructor

    SRHT(;
        cardinality = Left(),
        compression_dim::Int64=2,
        block_size::Int64=10,
        type::Type{N}=Float64
    ) where {N <: Number}

## Keywords
- `cardinality::Cardinality`, the direction the compression matrix is intended to be
    applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
    By default `Left()` is chosen.
- `compression_dim::Int64`, the target compression dimension. Referred to as ``s`` in the
    mathematical description. By default this is set to 2.
- `block_size::Int64`, the number of vectors in the padding matrix.
- `type::Type{<:Number}`, the type of elements in the compressor.

## Returns
- A `SRHT` object.

## Throws
- `ArgumentError` if `compression_dim` is non-positive or if `block_size` is non-positive.
"""
struct SRHT <: Compressor
    cardinality::Cardinality
    compression_dim::Int64
    block_size::Int64
    type::Type{<:Number}

    # perform checks on values of fields
    function SRHT(cardinality, compression_dim, block_size, type)
        if compression_dim <= 0
            throw(ArgumentError("Field `compression_dim` must be positive."))
        elseif block_size <= 0
            throw(ArgumentError("Field `block_size` must be positive."))
        elseif typeof(cardinality) == Undef
            throw(ArgumentError("Cardinality must be of type `Left` or `Right`."))
        end
            
        return new(cardinality, compression_dim, block_size, type)
    end

end

function SRHT(;
    cardinality = Right(),
    compression_dim::Int64=2,
    block_size::Int64=10,
    type::Type{N}=Float64
) where {N <: Number}
    return SRHT(cardinality, compression_dim, block_size, type)
end

"""
    SRHTRecipe{C<:Cardinality, M<:AbstractMatrix} <: CompressorRecipe

The recipe containing all allocations and information for the SRHT compressor.

# Fields
- `cardinality::Cardinality`, the direction the compression matrix is intended to
    be applied to a target matrix or operator. Values allowed are `Left()` or `Right()`.
- `n_rows::Int64`, the number of rows of the compression matrix.
- `n_cols::Int64`, the number of columns of the compression matrix.
- `scale::Float64`, the factor to ensure the isopmorphism of the sketch.
- `op::Vector{Int64}`, the vector of indices to be subsampled.
- `signs::BitVector`, the vector of signs where `0` indicates negative one and `1` indicates
    positive one. 
- `padding::AbstractMatrix`, the matrix containing the padding for the matrix being 
    sketched.

# Constructor
    SRHTRecipe(
        
"""
mutable struct SRHTRecipe{C<:Cardinality, M<:AbstractMatrix} <: CompressorRecipe
    cardinality::C
    n_rows::Int64
    n_cols::Int64
    scale::Float64
    op::Vector{Int64}
    signs::BitVector
    padding::M
end
