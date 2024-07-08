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
represents the CountSketch algorithm. The assumption is that A is fully known (that is not in a streaming context).

See Kenneth Clarkson and David Woodruff. "Low Rank Approximation and Regression in Input Sparsity Time"

# Fields

- `size::Int64` is the number of rows in the sketch matrix.

Calling `LinSysBlockRowCountSketch()` defaults to `LinSysBlockRowCountSketch(1)`.
"""

# constructor for CountSketch; requires the size of the sketch; default 1.
mutable struct LinSysBlkRowCountSketch <: LinSysVecRowSelect 
    size::Int64
    labels::Union{Array{Int64},Nothing}
    signs::Union{Array{Int64},Nothing}
    S::Union{Matrix{Int64},Nothing}
end

function LinSysBlkRowCountSketch(size::Int64)
    return LinSysBlkRowCountSketch(size,nothing,nothing,nothing)
end

LinSysBlkRowCountSketch() = LinSysBlkRowCountSketch(1)

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowCountSketch,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    # size checking
    n = size(A)[1]
    if iter == 1
        if type.size <= 0
            throw("Sketch size is 0 or negative!")
        end
        type.S = Matrix{Int64}(undef,type.size,n)
        type.labels = Array{Int64}(undef,n)
        type.signs = Array{Int64}(undef,n) 
    end

    # Assign labels to rows and potential sign flip
    # TODO: More efficient implementation using hash tables?
    fill!(type.S, 0)  
    rand!(type.labels, 1:type.size) 
    rand!(type.signs, [-1,1]) 
    @inbounds for j in 1:n
        type.S[abs(type.labels[j]),j] = type.signs[j]
    end
    
    # sketched matrix
    SA = type.S*A

    # residual of sketched system
    res = SA*x - type.S*b

    return type.S,SA,res
end

# TODO: sparse matrix implementation?