# This file is part of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports type
#
# Date: 07/05/2024
# Author: Christian Varner
# Purpose: Implement a row sketching algorithm called CountSketch.

"""
    LinSysBlockRowCountSketch <: LinSysVecRowSelect

A mutable structure with one field that indicates the sketch size, and
represents the CountSketch algorithm. The assumption is that A is fully known (that is not in a streaming context).

See Kenneth Clarkson and David Woodruff. "Low Rank Approximation and Regression in Input Sparsity Time"

# Fields

- `size::Int64` is the number of rows in the sketch matrix.

Calling `LinSysBlockRowCountSketch()` defaults to `LinSysBlockRowCountSketch(1)`.
"""

# constructor for CountSketch; requires the size of the sketch; default 1.
mutable struct LinSysBlockRowCountSketch <: LinSysVecRowSelect 
    size::Int64
end
LinSysBlockRowCountSketch() = LinSysBlockRowCountSketch(1) 

# Common sample interface for linear systems
function sample(
    type::LinSysBlockRowCountSketch,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    # size checking
    if iter == 1
        if size <= 0
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

# test if this is better of worse than before
function sample2(
    type::LinSysBlockRowCountSketch,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    # size checking
    if iter == 1
        if size <= 0
            throw("Sketch size is 0 or negative!")
        end
    end

    # Assign labels to rows and potential sign flip
    n = size(A)[1]
    labels = rand(1:type.size, n)
    hash_labels = Dict()

    rown = 1
    for lab in labels  
        if lab in keys(hash_labels)
            append!(hash_labels[lab], rown)
            rown += 1
        else
            hash_labels[lab] = [rown]
            rown += 1
        end
    end
    
    S = zeros(type.size, n)    
    for lab in keys(hash_labels)
        S[lab,hash_labels[lab]] = rand([-1,1],length(hash_labels[lab]))
    end

    SA = S*A
    return S, SA, SA*x - S*b
end