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

- `blockSize::Int64`, number of rows sampled (i.e., number of rows in `S * A`)
- `probability::Union{Weights, Vector{Float64}, Nothing}`, probability vector that is used to sample without replacement. Weights for each row of `A` when sampling.
- `population::Union{Vector{Int64}, Nothing}`, if number of rows are known, it is a vector containing the elements in the set {1,...,n}.
- `rowsSampled::Union{Vector{Int64}, Nothing}`, buffer array to hold index of rows sampled from `A`.
- `S::Union{Matrix{Int64}, Nothing}`, buffer array to hold sketched matrix.

Calling `LinSysBlkRowSelectWoReplacement()` defaults to `LinSysBlkRowSelectWoReplacement(1, nothing, nothing, nothing, nothing)`.

Three other constructors are provided besides the full one.

- `LinSysBlkRowSelectWoReplacement(blockSize::Int64, probability::Union{Weights, Vector{Float64}}, population::Vector{Int64})`
- `LinSysBlkRowSelectWoReplacement(blockSize::Int64, probability::Union{Weights,Vector{Float64}})`
- `LinSysBlkRowSelectWoReplacement(blockSize::Int64)` 
"""
mutable struct LinSysBlkRowSelectWoReplacement <: LinSysVecRowSelect
    blockSize::Int64
    probability::Union{Weights, Vector{Float64}, Nothing}
    population::Union{Vector{Int64}, Nothing}
    rowsSampled::Union{Vector{Int64}, Nothing}
    S::Union{Matrix{Int64}, Nothing}
end

# TODO: Constructors
function LinSysBlkRowSelectWoReplacement(blockSize::Int64, probability::Union{Weights, Vector{Float64}}, population::Vector{Int64})
    if isa(probability, Weights)
        return LinSysBlkRowSelectWoReplacement(blockSize, probability, population, nothing)
    else
        return LinSysBlkRowSelectWoReplacement(blockSize, Weights(probability), population, nothing) 
    end
end

function LinSysBlkRowSelectWoReplacement(blockSize::Int64, probability::Union{Weights,Vector{Float64}})
    if isa(probability, Weights)
        return LinSysBlkRowSelectWoReplacement(blockSize, probability, nothing, nothing)
    else
        return LinSysBlkRowSelectWoReplacement(blockSize, Weights(probability), nothing, nothing) 
    end
end

function LinSysBlkRowSelectWoReplacement(blockSize::Int64)
    return LinSysBlkRowSelectWoReplacement(blockSize, nothing, nothing, nothing, nothing)
end

LinSysBlkRowSelectWoReplacement() = LinSysBlkRowSelectWoReplacement(1)

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowSelectWoReplacement,
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
            throw(DomainError("blockSize is larger than the number of rows in A! Cannot select this amount without replacement!"))
        end

        # initialization buffer arrays
        type.S = Matrix{Int64}(undef, type.blockSize, n)
        type.rowsSampled = Vector{Int64}(undef, type.blockSize)

        # check struct data and initialize
        if isnothing(type.probability)
            type.probability = Weights(repeat([1/n],outer=n))
        elseif isa(type.probability,Vector)
            if sum(type.probability) != 1
                throw(DomainError("Weights do not sum to one!"))
            end
            type.probability = Weights(type.probability)
        end
        
        if isnothing(type.population)
            type.population = [i for i in 1:n]
        end
    end

    # Form the sketched matrix S
    fill!(type.S, 0)
    StatsBase.sample!(type.population, type.probability, type.rowsSampled, replace = false, ordered = false)
    for j in 1:type.blockSize
        type.S[j, type.rowsSampled[j]] = 1
    end

    # form sketched matrix `S * A`
    SA = A[type.rowsSampled, :]

    # form sketched residual
    res = SA * x - b[type.rowsSampled, :]

    return type.S, SA, res
end