# This file is part of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports type
#
# Date: 07/19/2024
# Author: Christian Varner
# Purpose: Implement the column sketching algorithm called CountSketch.

"""
    LinSysBlkColCountSketch <: LinSysVecColSelect

A mutable structure that represents the CountSketch algorithm for columns.
The assumption is that `A` is fully known (that is, the sampling procedure
is not used in a streaming context).

See Kenneth L. Clarkson and David P. Woodruff. 2017. 
    "Low-Rank Approximation and Regression in Input Sparsity Time."
    J. ACM 63, 6, Article 54 (February 2017), 45 pages. 
    https://doi.org/10.1145/3019134

# Fields

- `block_size::Int64`, is the number of columns in the sketched matrix `A * S`
- `S::Union{Matrix{Int64}, Nothing}`, buffer matrix for storing the sampling matrix `S`.

Additional Constructors:

Calling `LinSysBlkColCountSketch(block_size)` defaults to `LinSysBlkColCountSketch(block_size, nothing)`.
Calling `LinSysBlkColCountSketch()` defaults to `LinSysBlkColCountSketch(2, nothing)`. 

!!! Remark "Implementation Note"
    Current implementation does not take advantage of sparse matrix data structures or operations.
"""
mutable struct LinSysBlkColCountSketch <: LinSysVecColSelect
    block_size::Int64
    S::Union{Matrix{Int64}, Nothing}
end

function LinSysBlkColCountSketch(block_size::Int64)
    # check if block size is non-positive and throw error
    if block_size <= 0
        throw(DomainError("block_size is 0 or negative!"))
    end

    return LinSysBlkColCountSketch(block_size, nothing)
end

LinSysBlkColCountSketch() = LinSysBlkColCountSketch(2)

function sample(
    type::LinSysBlkColCountSketch,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    # block_size checking and initialization of memory
    ncolA = size(A)[2]
    if iter == 1
        if type.block_size <= 0
            throw(DomainError("block_size is 0 or negative!"))
        end

        if type.block_size > ncolA
            @warn("block_size is greater than the number of columns in A!")
        end

        type.S = zeros(Int64, ncolA, type.block_size)
    end

    # reset previous sketch matrix 
    if iter > 1 
        fill!(type.S, 0)
    end

    # for each row, assign a -1 or 1 randomly
    signs = [-1, 1]
    @inbounds for j in 1:ncolA
        type.S[j, rand(1:type.block_size)] = rand(signs) # (ncolA, block_size)
    end

    # form sketched matrix `A * S`
    AS = A * type.S

    # form full residual
    res = A * x - b

    # form sketched residual
    grad = AS' * res

    return type.S, AS, res, grad
end