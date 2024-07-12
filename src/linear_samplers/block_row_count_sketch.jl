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

A mutable structure that represents the CountSketch algorithm. 
The assumption is that `A` is fully known (that is, the sampling procedure is not used in a streaming context).

See Kenneth L. Clarkson and David P. Woodruff. 2017. 
    "Low-Rank Approximation and Regression in Input Sparsity Time."
    J. ACM 63, 6, Article 54 (February 2017), 45 pages. 
    https://doi.org/10.1145/3019134

# Fields

- `blockSize::Int64`, is the number of rows in the sketched matrix `S * A`.
- `S::Union{Matrix{Int64},Nothing}`, buffer matrix for storing the sampling matrix `S`.
- `signs::Union{Vector{Int64},Nothing}`, buffer vector for storing data used in `sample`

Calling `LinSysBlockRowCountSketch(blockSize)` defaults to `LinSysBlockRowCountSketch(blockSize, nothing, nothing)`.

Remark: Current implementation does not take advantage of sparse matrix data structures or operations.
"""

mutable struct LinSysBlkRowCountSketch <: LinSysVecRowSelect 
    blockSize::Int64
    S::Union{Matrix{Int64}, Nothing}
    signs::Union{Vector{Int64},Nothing}
end

# Additional constructor for LinSysBlkRowCountSketch
function LinSysBlkRowCountSketch(blockSize::Int64)
    return LinSysBlkRowCountSketch(blockSize, nothing, nothing)
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
            throw(DomainError("blockSize is 0 or negative!")) 
        end
        
        if type.blockSize > n
            @warn("blockSize is larger than the number of rows in A!")
        end

        # initializations
        type.S = Matrix{Int64}(undef, type.blockSize, n)
        type.signs = [-1, 1]
    end

    # form sketching matrix
    fill!(type.S, 0)  
    @inbounds for j in 1:n
        # assign labels to rows and possible sign flips 
        type.S[rand(1:type.blockSize),j] = rand(type.signs)
    end
    
    # form sketched matrix `S * A`
    SA = type.S * A

    # form sketched residual
    res = SA * x - type.S * b

    return type.S, SA, res
end