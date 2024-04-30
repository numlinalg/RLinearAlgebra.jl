
# This file is part of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

# using Random

"""
    LinSysVecRowBlockGaussian <: LinSysVecColSelect

A mutable structure with fields to handle Guassian column sketching. 

# Fields
- `blockSize::Int64` - Specifies the size of each block.
- `sketchMatrix::Union{AbstractMatrix, Nothing}` - The buffer for storing the Gaussian sketching matrix.
- `scaling::Float64` - The variance of the sketch, is set to be s/n.

Calling `LinSysVecRowBlockGaussian()` defaults to setting `blockSize` to 2.
"""
mutable struct LinSysVecRowBlockGaussian <: LinSysVecColSelect
    blockSize::Int64
    sketchMatrix::Union{AbstractMatrix, Nothing}
    scaling::Float64
end

LinSysVecRowBlockGaussian(blockSize) = LinSysVecRowBlockGaussian(blockSize, nothing)
LinSysVecRowBlockGaussian() = LinSysVecRowBlockGaussian(2, nothing)

# Common sample interface for linear systems
function sample(
    type::LinSysVecRowBlockGaussian,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    if iter == 1
        m, n = size(A)
        scaling = sqrt(type.blockSize / m)
        type.SketchMatrix = Matrix{Float64}(s, m) 
    end
    randn!(type.SketchMatrix)
    type.SketchMatrix .*= type.scaling
    SA = type.SketchMatrix * A
    Sb = type.SketchMatrix * b
    # Residual of the linear system
    res = SA * x - Sb

    return type.SketchMatrix, SA, res, sb
end
