# This code was Written by Nathaniel Pritchard
"""
    LinSysBlkRowGaussSampler <: LinSysBlkRowSampler

A mutable structure with fields to handle Guassian row sketching. 

# Fields
- `blockSize::Int64` - Specifies the size of each block.
- `sketchMatrix::Union{AbstractMatrix, Nothing}` - The buffer for storing the Gaussian sketching matrix.
- `scaling::Float64` - The variance of the sketch, is set to be s/n.

Calling `LinSysBlkRowGaussSampler()` defaults to setting `blockSize` to 2.
"""
mutable struct LinSysBlkRowGaussSampler <: LinSysBlkRowSampler
    blockSize::Int64
    sketchMatrix::Union{AbstractMatrix, Nothing}
    scaling::Float64
end

LinSysBlkRowGaussSampler(blockSize) = LinSysBlkRowGaussSampler(blockSize, nothing, 0.0)
LinSysBlkRowGaussSampler() = LinSysBlkRowGaussSampler(2, nothing, 0.0)

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowGaussSampler,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    if iter == 1
        m, n = size(A)
        # Compute scaling for matrix
        type.scaling = sqrt(type.blockSize / m)
        # Allocate for the sketching matrix
        type.sketchMatrix = Matrix{Float64}(undef, type.blockSize, m) 
    end
    
    # Fill the allocated matrix with N(0,1) entries
    randn!(type.sketchMatrix)
    # Scale the matrix to ensure it has expectation 1 when applied to unit vector
    type.sketchMatrix .*= type.scaling
    SA = type.sketchMatrix * A
    Sb = type.sketchMatrix * b
    # Residual of the linear system
    res = SA * x - Sb
    return type.sketchMatrix, SA, res
end
