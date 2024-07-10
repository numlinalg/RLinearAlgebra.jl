# This file is part of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports type
#
# Date: 07/05/2024
# Author: Christian Varner
# Purpose: Implement a row sketching algorithm called CountSketch.

"""
    LinSysBlkRowCountSketch <: LinSysVecRowSelect

An mutable structure with one field that indicates the sketch size, and
represents the CountSketch algorithm. The assumption is that `A`` is fully known (that is, 
the sampling procedure is not used in a streaming context).

See Kenneth L. Clarkson and David P. Woodruff. 2017. 
    "Low-Rank Approximation and Regression in Input Sparsity Time."
    J. ACM 63, 6, Article 54 (February 2017), 45 pages. 
    https://doi.org/10.1145/3019134

# Fields

- `blockSize::Int64`, is the number of rows in `SA`.
- `labels::Union{Array{Int64},Nothing}`, buffer array that stores data for sketching. 
- `signs::Union{Array{Int64},Nothing}`, buffer array that stores data for sketching.
- `S::Union{Matrix{Int64},Nothing}`, buffer matrix for storing sketched matrix.

Calling `LinSysBlockRowCountSketch(blockSize)` defaults to `LinSysBlockRowCountSketch(blockSize, nothing, nothing, nothing)`.
"""

mutable struct LinSysBlkRowCountSketch <: LinSysVecRowSelect 
    blockSize::Int64
    labels::Union{Array{Int64}, Nothing}
    signs::Union{Array{Int64}, Nothing}
    S::Union{Matrix{Int64}, Nothing}
end

# Additional constructors for LinSysBlkRowCountSketch
function LinSysBlkRowCountSketch(blockSize::Int64)
    return LinSysBlkRowCountSketch(blockSize, nothing, nothing, nothing)
end

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowCountSketch,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    # blockSize checking and initialization of memory
    n = size(A)[1]
    if iter == 1
        if type.blockSize <= 0
            throw("blockSize is 0 or negative!")
        end
        type.S = Matrix{Int64}(undef, type.blockSize, n)
        type.labels = Array{Int64}(undef, n)
        type.signs = Array{Int64}(undef, n) 
    end

    # Assign labels to rows and generate possible flip signs
    fill!(type.S, 0)  
    rand!(type.labels, 1:type.blockSize) 
    rand!(type.signs, [-1,1])
    
    # form sketching matrix
    @inbounds for j in 1:n
        type.S[type.labels[j], j] = type.signs[j]
    end
    
    # sketched matrix
    SA = type.S * A

    # residual of sketched linear system
    res = SA * x - type.S * b

    return type.S, SA, res
end