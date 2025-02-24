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
    LinSysBlkRowSelectWoReplacement <: LinSysBlkRowSampler

A mutable struct that represents sampling rows from `A` without replacement using an
arbitrary weight/probability vector.

# Fields

- `block_size::Int64`, number of rows sampled (i.e., number of rows in `S * A`).
- `probability::Union{Weights, Vector{Float64}, Nothing}`, vector that represents a probability distribution over the rows of `A`. Requirements are that the probabilities sum to 1, are non-negative, `probability` has the same length as number of rows in `A`, and `probability` has at least as many positive entries as `block_size`. If `probability` is unspecified in the constructor, `sample` will default to a uniform distribution over rows of `A`.
- `population::Union{Vector{Int64}, Nothing}`, buffer array to hold `collect(1:size(A)[1])` used in `sample`.
- `rows_sampled::Union{Vector{Int64}, Nothing}`, buffer array to hold index of rows sampled from `A`.
- `S::Union{Matrix{Int64}, Nothing}`, buffer array to hold sketched matrix `S`.

Calling `LinSysBlkRowSelectWoReplacement()` defaults to `LinSysBlkRowSelectWoReplacement(2, nothing, nothing, nothing, nothing)`.

An additional constructor is provided with keyword arguments `block_size` and `probability`.
"""
mutable struct LinSysBlkRowSelectWoReplacement <: LinSysBlkRowSampler
    block_size::Int64
    probability::Union{Weights, Vector{Float64}, Nothing}
    population::Union{Vector{Int64}, Nothing}
    rows_sampled::Union{Vector{Int64}, Nothing}
    S::Union{Matrix{Int64}, Nothing}
    LinSysBlkRowSelectWoReplacement(block_size, probability, population, rows_sampled, S) = begin
        check_properties = (block_size > 0)
        @assert check_properties "block_size is 0 or negative."
        return new(block_size, probability, population, rows_sampled, S)
    end
end

function LinSysBlkRowSelectWoReplacement(;block_size=2, probability=nothing)
    return LinSysBlkRowSelectWoReplacement(block_size, probability, nothing, nothing, nothing)
end

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowSelectWoReplacement,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    # Error checking and initialization on first iteration
    if iter == 1
        nrow = size(A)[1]
        if type.block_size > nrow
            throw(DomainError("block_size is larger than the number of rows in A! Cannot select this amount without replacement!"))
        end

        # initialization buffer arrays
        type.S = Matrix{Int64}(undef, type.block_size, nrow)
        type.rows_sampled = Vector{Int64}(undef, type.block_size)

        # error checking on probability vector, possible initialization and type conversion
        if isnothing(type.probability)
            type.probability = Weights(repeat([1/nrow], outer = nrow))
        elseif isa(type.probability, Vector) || isa(type.probability, Weights)
            # check that probability is a valid distribution on the rows 
            if !(sum(type.probability) ≈ 1)
                throw(DomainError("Elements of `probability` do not sum to 1!"))
            elseif sum(type.probability .>= 0) != size(type.probability)[1]
                throw(DomainError("Not all probabilities are non-negative in `probability`!")) 
            elseif size(type.probability)[1] != size(A)[1]
                throw(DimensionMismatch("Length of `probability` vector is smaller than the number of rows in A!"))
            elseif sum(type.probability .> 0) < type.block_size
                throw(DimensionMismatch("Not enough non-zero probabilities in `probability` to select the required number of rows!"))
            end
            
            # type conversion if necessary
            if isa(type.probability, Vector)
                type.probability = Weights(type.probability)
            end
        end
        
        # form the rows to sample from
        type.population = collect(1:nrow)
    end

    # form the sketched matrix S
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
