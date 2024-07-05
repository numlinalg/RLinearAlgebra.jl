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

A mutable structure with one field that indicates the sketch size, and
represents the CountSketch algorithm. The assumption is that A is fully known (that is not in a streaming context).

See Kenneth Clarkson and David Woodruff. "Low Rank Approximation and Regression in Input Sparsity Time"

# Fields

- `size::Int64` is the number of rows in the sketch matrix.

Calling `LinSysBlockRowCountSketch()` defaults to `LinSysBlockRowCountSketch(1)`.
"""

# constructor for CountSketch; requires the size of the sketch; default 1.
mutable struct LinSysBlkRowCountSketch <: LinSysVecRowSelect 
    size::Int64
end
LinSysBlockRowCountSketch() = LinSysBlockRowCountSketch(1) 

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowCountSketch,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    # size checking
    if iter == 1
        if type.size <= 0
            throw("Sketch size is 0 or negative!")
        end
    end

    # Assign labels to rows and potential sign flip
    # TODO: More efficient implementation using hash tables?
    n = size(A)[1]
    S = zeros(type.size, n)   
    for j in 1:n
        S[rand(1:type.size),j] = rand([-1,1])
    end

    SA = S*A
    return S, SA, SA*x - S*b
end