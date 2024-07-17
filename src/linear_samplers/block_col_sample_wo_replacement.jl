# This file is part of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports type
#
# Date: 07/14/2024
# Author: Christian Varner
# Purpose: Implement a col sketching algorithm that selects
# columns without replacement from an arbitrary probability distribution.

"""
    LinSysBlkColSelectWoReplacement <: LinSysVecColSelect

A mutable struct that represents sampling columns without replacement using an
arbitrary weight/probability vector.

# Fields

- `block_size::Int64`, number of columns sampled (i.e., number of columns in `S * A`)
- `probability::Union{Weights, Vector{Float64}, Nothing}`, probability vector that is used to sample without replacement. Weights for each column of `A` when sampling.
- `population::Union{Vector{Int64}, Nothing}`, if number of columns are known, it is a vector containing the elements in the set {1,...,number of columns of `A`}.
- `col_sampled::Union{Vector{Int64}, Nothing}`, buffer array to hold index of columns sampled from `A`.
- `S::Union{Matrix{Int64}, Nothing}`, buffer array to hold sketched matrix.

Calling `LinSysBlkColSelectWoReplacement()` defaults to `LinSysBlkColSelectWoReplacement(2, nothing, nothing, nothing, nothing)`.

An additional constructor is provided with the keyword arguments `block_size` and `probability`
"""
mutable struct LinSysBlkColSelectWoReplacement <: LinSysVecColSelect
    block_size::Int64
    probability::Union{Weights, Vector{Float64}, Nothing}
    population::Union{Vector{Int64}, Nothing}
    col_sampled::Union{Vector{Int64}, Nothing}
    S::Union{Matrix{Int64}, Nothing}
end

function LinSysBlkColSelectWoReplacement(;block_size = 2, probability = nothing)
    return LinSysBlkColSelectWoReplacement(block_size, probability, nothing, nothing, nothing)
end

LinSysBlkColSelectWoReplacement() = LinSysBlkColSelectWoReplacement(block_size = 2)

# Common sample interface for linear systems
function sample(
    type::LinSysBlkColSelectWoReplacement,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    # block_size checking and initialization of memory
    ncol = size(A)[2]
    if iter == 1
        if type.block_size <= 0
            throw(DomainError("block_size is 0 or negative!")) 
        end
        
        if type.block_size > ncol
            throw(DomainError("block_size is larger than the number of columns in A! Cannot select this amount without replacement!"))
        end

        # initialization buffer arrays
        type.S = Matrix{Int64}(undef, ncol, type.block_size)
        type.col_sampled = Vector{Int64}(undef, type.block_size)

        # check struct data and initialize
        if isnothing(type.probability)
            type.probability = Weights(repeat([1/ncol],outer=ncol))
        elseif isa(type.probability, Vector) || isa(type.probability, Weights)
            # check that probability is a valid distribution on the rows
            if sum(type.probability) != 1
                throw(DomainError("Weights do not sum to one!"))
            elseif sum(type.probability .>= 0) != size(type.probability)[1]
                throw(DomainError("Not all probabilities are non-negative in probability!"))
            elseif size(type.probability)[1] != size(A)[2]
                throw(DimensionMismatch("probability vector is smaller than the number of columns!"))
            elseif sum(type.probability .> 0) < type.block_size
                throw(DimensionMismatch("Not enough non-zero probabilities in probability to select the required number of blocks!"))
            end
            
            # type conversion if necessary
            if isa(type.probability, Vector)
                type.probability = Weights(type.probability)
            end
        end
        
        # form the columns to sample from
        type.population = collect(1:ncol)
    end

    # Form the sketched matrix S
    fill!(type.S, 0)
    StatsBase.sample!(type.population, type.probability, type.col_sampled, replace = false, ordered = false)
    @inbounds for j in 1:type.block_size
        type.S[type.col_sampled[j],j] = 1
    end

    # form sketched matrix `S * A`
    AS = A[:, type.col_sampled] # dim n*blocksize

    # form full residual
    res = A * x - b # dim n

    # formm sketched residual
    grad = AS' * res

    return type.S, AS, res, grad
end