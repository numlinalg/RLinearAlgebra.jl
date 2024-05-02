
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
mutable struct LinSysVecRowBlockGaussian <: LinSysVecRowSelect
    blockSize::Int64
    sketchMatrix::Union{AbstractMatrix, Nothing}
    scaling::Float64
end

LinSysVecRowBlockGaussian(blockSize) = LinSysVecRowBlockGaussian(blockSize, nothing, 0.0)
LinSysVecRowBlockGaussian() = LinSysVecRowBlockGaussian(2, nothing, 0.0)

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
        type.scaling = sqrt(type.blockSize / m)
        type.sketchMatrix = Matrix{Float64}(undef, type.blockSize, m) 
    end

    randn!(type.sketchMatrix)
    type.sketchMatrix .*= type.scaling
    SA = type.sketchMatrix * A
    Sb = type.sketchMatrix * b
    # Residual of the linear system
    res = SA * x - Sb
    return type.sketchMatrix, SA, res
end
