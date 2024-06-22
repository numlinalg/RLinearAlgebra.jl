# This code was written by Nathaniel Pritchard
"""
    LinSysBlkColGaussSampler <: LinSysBlkColSampler 

A mutable structure with fields to handle Guassian column sketching. 

# Fields
- `blockSize::Int64` - Specifies the size of each block.
- `sketchMatrix::Union{AbstractMatrix, Nothing}` - The buffer for storing the Gaussian sketching matrix.
- `scaling::Float64` - The variance of the sketch, is set to be blockSize/numberOfColumns.

Calling `LinSysBlkColGaussSampler()` defaults to setting `blockSize` to 2.
"""
mutable struct LinSysBlkColGaussSampler <: LinSysBlkColSampler 
    blockSize::Int64
    sketchMatrix::Union{AbstractMatrix, Nothing}
    scaling::Float64
end

LinSysBlkColGaussSampler(blockSize) = LinSysBlkColGaussSampler(blockSize, nothing, 0.)
LinSysBlkColGaussSampler() = LinSysBlkColGaussSampler(2, nothing, 0.)

# Common sample interface for linear systems
function sample(
    type::LinSysBlkColGaussSampler,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    if iter == 1
        m, n = size(A)
        type.scaling = sqrt(type.blockSize / n)
        type.sketchMatrix = Matrix{Float64}(undef, n, type.blockSize) 
    end

    randn!(type.sketchMatrix)
    type.sketchMatrix .*= type.scaling
    SA = A * type.sketchMatrix
    # Residual of the linear system
    res = A * x - b
    # Normal equation residual in the Sketched Block
    grad = SA' * (A * x - b)
    return type.sketchMatrix, SA, grad, res
end

