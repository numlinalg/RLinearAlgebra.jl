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

- `blockSize::Int64`, number of columns sampled (i.e., number of columns in `S * A`)
- `probability::Union{Weights, Vector{Float64}, Nothing}`, probability vector that is used to sample without replacement. Weights for each column of `A` when sampling.
- `population::Union{Vector{Int64}, Nothing}`, if number of columns are known, it is a vector containing the elements in the set {1,...,number of columns of `A`}.
- `rowsSampled::Union{Vector{Int64}, Nothing}`, buffer array to hold index of columns sampled from `A`.
- `S::Union{Matrix{Int64}, Nothing}`, buffer array to hold sketched matrix.

Calling `LinSysBlkColSelectWoReplacement()` defaults to `LinSysBlkColSelectWoReplacement(1, nothing, nothing, nothing, nothing)`.

Three other constructors are provided besides the full one.

- `LinSysBlkColSelectWoReplacement(blockSize::Int64, probability::Union{Weights, Vector{Float64}}, population::Vector{Int64})`
- `LinSysBlkColSelectWoReplacement(blockSize::Int64, probability::Union{Weights,Vector{Float64}})`
- `LinSysBlkColSelectWoReplacement(blockSize::Int64)` 
"""
mutable struct LinSysBlkColSelectWoReplacement <: LinSysVecColSelect
    blockSize::Int64
    probability::Union{Weights, Vector{Float64}, Nothing}
    population::Union{Vector{Int64}, Nothing}
    colSampled::Union{Vector{Int64}, Nothing}
    S::Union{Matrix{Int64}, Nothing}
end

# TODO: Constructors
function LinSysBlkColSelectWoReplacement(blockSize::Int64, probability::Union{Weights, Vector{Float64}}, population::Vector{Int64})
    if isa(probability, Weights)
        return LinSysBlkColSelectWoReplacement(blockSize, probability, population, nothing)
    else
        return LinSysBlkColSelectWoReplacement(blockSize, Weights(probability), population, nothing) 
    end
end

function LinSysBlkColSelectWoReplacement(blockSize::Int64, probability::Union{Weights,Vector{Float64}})
    if isa(probability, Weights)
        return LinSysBlkColSelectWoReplacement(blockSize, probability, nothing, nothing)
    else
        return LinSysBlkColSelectWoReplacement(blockSize, Weights(probability), nothing, nothing) 
    end
end

function LinSysBlkColSelectWoReplacement(blockSize::Int64)
    return LinSysBlkColSelectWoReplacement(blockSize, nothing, nothing, nothing, nothing)
end

LinSysBlkColSelectWoReplacement() = LinSysBlkColSelectWoReplacement(1)

# Common sample interface for linear systems
function sample(
    type::LinSysBlkColSelectWoReplacement,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)

    # blockSize checking and initialization of memory
    d = size(A)[2]
    if iter == 1
        if type.blockSize <= 0
            throw(DomainError("blockSize is 0 or negative!")) 
        end
        
        if type.blockSize > d
            throw(DomainError("blockSize is larger than the number of columns in A! Cannot select this amount without replacement!"))
        end

        # initialization buffer arrays
        type.S = Matrix{Int64}(undef, d, type.blockSize)
        type.colSampled = Vector{Int64}(undef, type.blockSize)

        # check struct data and initialize
        if isnothing(type.probability)
            type.probability = Weights(repeat([1/d],outer=d))
        elseif isa(type.probability,Vector)
            if sum(type.probability) != 1
                throw(DomainError("Weights do not sum to one!"))
            elseif len(type.probability) != d
                throw(DomainError("Not all columns of weights. Add weights to probability."))
            end
            type.probability = Weights(type.probability)
        end
        
        if isnothing(type.population)
            type.population = [i for i in 1:d]
        end
    end

    # Form the sketched matrix S
    fill!(type.S, 0)
    StatsBase.sample!(type.population, type.probability, type.colSampled, replace = false, ordered = false)
    for j in 1:type.blockSize
        type.S[type.colSampled[j],j] = 1
    end

    # form sketched matrix `S * A`
    AS = A[:, type.colSampled] # dim n*blocksize

    # form full residual
    res = A * x - b # dim n

    # from sketched residual
    grad = AS' * res

    return type.S, AS, res, grad
end