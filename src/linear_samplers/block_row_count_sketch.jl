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

A mutable structure that represents the CountSketch algorithm for rows. 
The assumption is that `A` is fully known (that is, the sampling procedure is not used in a streaming context).

See Kenneth L. Clarkson and David P. Woodruff. 2017. 
    "Low-Rank Approximation and Regression in Input Sparsity Time."
    J. ACM 63, 6, Article 54 (February 2017), 45 pages. 
    https://doi.org/10.1145/3019134

The explicit sketch matrix is mentioned in Section 1.2 - Techniques (row version) of the aforementioned paper.
See also https://wangshusen.github.io/code/countsketch.html for a visual explanation of the column version.

# Fields

- `block_size::Int64`, is the number of rows in the sketched matrix `S * A`.
- `S::Union{Matrix{Int64}, Nothing}`, buffer matrix for storing the sampling matrix `S`.

Additional Constructors:

Calling `LinSysBlkRowCountSketch(block_size)` defaults to `LinSysBlkRowCountSketch(block_size, nothing)`.
Calling `LinSysBlkRowCountSketch()` defaults to `LinSysBlkRowCountSketch(2, nothing)`. 

!!! Remark "Implementation Note" 
    Current implementation does not take advantage of sparse matrix data structures or operations.
"""
mutable struct LinSysBlkRowCountSketch <: LinSysVecRowSelect 
    block_size::Int64
    S::Union{Matrix{Int64}, Nothing}
end

# Additional constructor for LinSysBlkRowCountSketch to specify just block_size
function LinSysBlkRowCountSketch(block_size::Int64)
    # check if block size is non-positive and throw error
    if block_size <= 0
        throw(DomainError("block_size is 0 or negative!"))
    end

    return LinSysBlkRowCountSketch(block_size, nothing)
end

# Default constructor for LinSysBlkRowCountSketch
LinSysBlkRowCountSketch() = LinSysBlkRowCountSketch(2)

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowCountSketch,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    # block_size checking and initialization of memory
    nrowA = size(A)[1]
    if iter == 1
        if type.block_size <= 0
            throw(DomainError("block_size is 0 or negative!")) 
        end
        
        if type.block_size > nrowA
            @warn("block_size is larger than the number of rows in A!")
        end

        # initializations
        type.S = zeros(Int64, type.block_size, nrowA)
    end

    # reset previous sketch matrix
    if iter > 1
        fill!(type.S, 0) 
    end 
    
    # for each column assign a row with a rand -1 or 1
    signs = [-1, 1] 
    @inbounds for j in 1:nrowA
        type.S[rand(1:type.block_size), j] = rand(signs) # (blocksize, size(A)[1])
    end
    
    # form sketched matrix `S * A`
    SA = type.S * A

    # form sketched residual
    res = SA * x - type.S * b

    return type.S, SA, res
end