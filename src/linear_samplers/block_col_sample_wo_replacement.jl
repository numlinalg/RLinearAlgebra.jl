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
    LinSysBlkColSelectWoReplacement <: LinSysBlkColSampler

A mutable struct that represents sampling columns from `A` without replacement using an
arbitrary weight/probability vector.

# Fields

- `block_size::Int64`, number of columns sampled (i.e., number of columns in `A * S`)
- `probability::Union{Weights, Vector{Float64}, Nothing}`, vector that represents a probability distribution over the columns of `A`. Requirements are that the probabilities sum to 1, are non-negative, `probability` has the same length as number of columns in `A`, and `probability` has at least as many positive entries as `block_size`. If `probability` is unspecified in the constructor, `sample` will default to a uniform distribution over columns of `A`.
- `population::Union{Vector{Int64}, Nothing}`, buffer array to hold the vector `collect(1:size(A)[2])` in the `sample` function.
- `col_sampled::Union{Vector{Int64}, Nothing}`, buffer array to hold index of columns sampled from `A`.
- `S::Union{Matrix{Int64}, Nothing}`, buffer array to hold sketching matrix `S`.

Calling `LinSysBlkColSelectWoReplacement()` defaults to `LinSysBlkColSelectWoReplacement(2, nothing, nothing, nothing, nothing)`.

An additional constructor is provided with the keyword arguments `block_size` and `probability`.
"""
mutable struct LinSysBlkColSelectWoReplacement <: LinSysBlkColSampler
    block_size::Int64
    probability::Union{Weights, Vector{Float64}, Nothing}
    population::Union{Vector{Int64}, Nothing}
    col_sampled::Union{Vector{Int64}, Nothing}
    S::Union{Matrix{Int64}, Nothing}
    LinSysBlkColSelectWoReplacement(block_size, probability, population, col_sampled, S) = begin
        check_properties = (block_size > 0)
        @assert check_properties "block_size is 0 or negative."
        return new(block_size, probability, population, col_sampled, S)
    end
end

function LinSysBlkColSelectWoReplacement(;block_size = 2, probability = nothing)
    return LinSysBlkColSelectWoReplacement(block_size, probability, nothing, nothing, nothing)
end

# Common sample interface for linear systems
function sample(
    type::LinSysBlkColSelectWoReplacement,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    # Error checking and initialization on first iteration
    if iter == 1
        ncol = size(A)[2]
        if type.block_size > ncol
            throw(DomainError("block_size is larger than the number of columns in A! Cannot select this amount without replacement!"))
        end

        # initialization buffer arrays
        type.S = Matrix{Int64}(undef, ncol, type.block_size)
        type.col_sampled = Vector{Int64}(undef, type.block_size)

        # error checking on probability vector, possible initialization and type conversion
        if isnothing(type.probability)
            type.probability = Weights(repeat([1/ncol], outer = ncol))
        elseif isa(type.probability, Vector) || isa(type.probability, Weights)
            # check that probability is a valid distribution on the rows
            if !(sum(type.probability) ≈ 1)
                throw(DomainError("Elements in `probability` do not sum to one!"))
            elseif sum(type.probability .>= 0) != size(type.probability)[1]
                throw(DomainError("Not all probabilities are non-negative in `probability`!"))
            elseif size(type.probability)[1] != size(A)[2]
                throw(DimensionMismatch("Length of `probability` vector is smaller than the number of columns in A!"))
            elseif sum(type.probability .> 0) < type.block_size
                throw(DimensionMismatch("Not enough non-zero probabilities in `probability` to select the required number of columns! Either decrease block_size, or reweight probability."))
            end
            
            # type conversion if necessary
            if isa(type.probability, Vector)
                type.probability = Weights(type.probability)
            end
        end
        
        # form the columns to sample from
        type.population = collect(1:ncol)
    end

    # form the sketched matrix S
    fill!(type.S, 0)
    StatsBase.sample!(type.population, type.probability, type.col_sampled, replace = false, ordered = false)
    @inbounds for j in 1:type.block_size
        type.S[type.col_sampled[j],j] = 1
    end

    # form sketched matrix `A * S`
    AS = A[:, type.col_sampled] # dim n*blocksize

    # form full residual
    res = A * x - b # dim n

    # form sketched residual
    grad = AS' * res

    return type.S, AS, res, grad
end
