
# This file is part of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type

# using Random

"""
    LinSysVecColBlockGaussian <: LinSysVecColSelect

A mutable structure with fields to handle Guassian column sketching. 

# Fields
- `blockSize::Int64` - Specifies the size of each block.
- `sketchMatrix::Union{AbstractMatrix, Nothing}` - The buffer for storing the Gaussian sketching matrix.
- `scaling::Float64` - The variance of the sketch, is set to be s/n.

Calling `LinSysVecColBlockGaussian()` defaults to setting `blockSize` to 2.
"""
mutable struct LinSysVecColBlockGaussian <: LinSysVecColSelect
    blockSize::Int64
    sketchMatrix::Union{AbstractMatrix, Nothing}
    scaling::Float64
end

LinSysVecColBlockGaussian(blockSize) = LinSysVecColBlockGaussian(blockSize, nothing)
LinSysVecColBlockGaussian() = LinSysVecColBlockGaussian(2, nothing)

# Common sample interface for linear systems
function sample(
    type::LinSysVecColBlockGaussian,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    if iter == 1
        m, n = size(A)
        scaling = sqrt(type.blockSize / n)
        type.SketchMatrix = Matrix{Float64}(n, s) 
    end
    randn!(type.SketchMatrix)
    type.SketchMatrix .*= type.scaling
    AS = A * type.SketchMatrix

    # Residual of the linear system
    res = A * x - b
    # Normal equation residual in the Sketched Block
    grad = AS' * (A * x - b)

    return type.SketchMatrix, AS, grad, res
end

#Function to update the solution 
function update_sol!(x::AbstractVector, update::AbstractVector, S::Matrix{Float64}, α::Real)
    x .-= α .* S' * update
end
#export LinSysVecBlockRandCyclic
