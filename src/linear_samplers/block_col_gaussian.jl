# This code was written by Nathaniel Pritchard
"""
    LinSysBlkColGaussSampler <: LinSysBlkColSampler 

A mutable structure with fields to handle Guassian column sketching. 

# Fields
- `block_size::Int64` - Specifies the size of each block.
- `sketch_matrix::Union{AbstractMatrix, Nothing}` - The buffer for storing the Gaussian sketching matrix.
- `scaling::Float64` - The variance of the sketch, is set to be block_size/numberOfColumns.

Calling `LinSysBlkColGaussSampler()` defaults to setting `block_size` to 2.
"""
mutable struct LinSysBlkColGaussSampler <: LinSysBlkColSampler 
    block_size::Int64
    sketch_matrix::Union{AbstractMatrix, Nothing}
    scaling::Float64
end

LinSysBlkColGaussSampler(block_size) = LinSysBlkColGaussSampler(block_size, nothing, 0.)
LinSysBlkColGaussSampler() = LinSysBlkColGaussSampler(2, nothing, 0.)

# Common sample interface for linear systems
function sample(
    type::LinSysBlkColGaussSampler,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    m, n = size(A)
    if iter == 1
        @assert type.block_size <= n "Block size must be less than col dimension"
        type.scaling = sqrt(type.block_size / n)
        type.sketch_matrix = Matrix{Float64}(undef, n, type.block_size) 
    end

    # Generate new sketch
    randn!(type.sketch_matrix)
    # Scale the matrix so it has expectation 1 when applied to unit vector
    type.sketch_matrix .*= type.scaling
    SA = A * type.sketch_matrix
    # Residual of the linear system
    res = A * x - b
    # Normal equation residual in the Sketched Block
    grad = SA' * (A * x - b)
    return type.sketch_matrix, SA, grad, res
end

