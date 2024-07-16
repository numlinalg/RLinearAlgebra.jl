# This file is part of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports type
#
# Date: 07/13/2024
# Author: Christian Varner
# Purpose: Implement a row sketching algorithm that selects
# rows without replacement from an arbitrary probability distribution.

"""
    LinSysBlkRowSelectWoReplacement <: LinSysVecRowSelect

A mutable struct that represents sampling rows without replacement using an
arbitrary weight/probability vector.

# Fields

- `block_size::Int64`, number of rows sampled (i.e., number of rows in `S * A`)
- `probability::Union{Weights, Vector{Float64}, Nothing}`, probability vector that is used to sample without replacement. Weights for each row of `A` when sampling.
- `population::Union{Vector{Int64}, Nothing}`, buffer array for data used in `sample`.
- `rows_sampled::Union{Vector{Int64}, Nothing}`, buffer array to hold index of rows sampled from `A`.
- `S::Union{Matrix{Int64}, Nothing}`, buffer array to hold sketched matrix `S`.

Calling `LinSysBlkRowSelectWoReplacement()` defaults to `LinSysBlkRowSelectWoReplacement(2, nothing, nothing, nothing, nothing)`.

An additional constructor is provided with keyword arguments `block_size, probability, population`.
"""
mutable struct LinSysBlkRowSelectWoReplacement <: LinSysVecRowSelect
    block_size::Int64
    probability::Union{Weights, Vector{Float64}, Nothing}
    population::Union{Vector{Int64}, Nothing}
    rows_sampled::Union{Vector{Int64}, Nothing}
    S::Union{Matrix{Int64}, Nothing}
end

# TODO: additional error checking

# Additional constructors
function LinSysBlkRowSelectWoReplacement(;block_size=2, probability=nothing)
    # perform type conversion
    if isa(probability, Vector)
        if sum(probability) != 1 || sum(probability .>= 0) != size(probability)[1]
            throw(DomainError("probability does not sum to 1, or elements of probability are not all non-negative"))
        end
        probability = Weights(probability)
    end

    # return struct
    return LinSysBlkRowSelectWoReplacement(block_size, probability, nothing, nothing, nothing)
end

# TODO: additional error checking

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowSelectWoReplacement,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    # block_size checking and initialization of memory
    nrow = size(A)[1]
    if iter == 1
        if type.block_size <= 0
            throw(DomainError("block_size is 0 or negative!")) 
        end
        
        if type.block_size > nrow
            throw(DomainError("block_size is larger than the number of rows in A! Cannot select this amount without replacement!"))
        end

        # initialization buffer arrays
        type.S = Matrix{Int64}(undef, type.block_size, nrow)
        type.rows_sampled = Vector{Int64}(undef, type.block_size)

        # check struct data and initialize
        if isnothing(type.probability)
            type.probability = Weights(repeat([1/nrow], outer=nrow))
        elseif isa(type.probability, Vector) 
            if sum(type.probability) != 1 || sum(probability .>= 0) != size(probability)[1]
                throw(DomainError("probability does not sum to 1, or elements of probability are not all non-negative"))
            end
            type.probability = Weights(type.probability)
        end
        
        type.population = collect(1:nrow)
    end

    # Form the sketched matrix S
    fill!(type.S, 0)
    StatsBase.sample!(type.population, type.probability, type.rows_sampled, replace = false, ordered = false) # sample rows wo replacement
    @inbounds for j in 1:type.block_size
        type.S[j, type.rows_sampled[j]] = 1
    end

    # form sketched matrix `S * A`
    SA = A[type.rows_sampled, :]

    # form sketched residual
    res = SA * x - b[type.rows_sampled, :]

    return type.S, SA, res
end