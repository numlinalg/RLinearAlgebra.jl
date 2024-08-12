# Date: 08/01/2024
# Author: Christian Varner
# Purpose: Parametric types for distributions.

##############################################################################
## This file is part of RLinearAlgebra.jl
##
## Overview: abstractions and interface methods for creating a distribution 
## over rows and columns of a matrix. 
##
## Contents
## - Abstract (parametric) Types
## - `initialize!` function documentation and interface
## - `getDistribution!` function documentation and interface.
## - Include statements
##
##############################################################################

"""
"""
abstract type SketchDirection end 

"""
"""
abstract type Left <: SketchDirection end # comes from rows
rows = Left

"""
"""
abstract type Right <: SketchDirection end # comes from columns
columns = Right

"""
    Distribution{T <: SketchDirection}

Parametric type that represents a distribution over the rows when
`T <: Left`, and over the columns when `T <: Right`. Corresponds
to left and right sketching of a matrix.
"""
abstract type Distribution{T <: SketchDirection} end

"""
    Base.eltype(dist::Type{Distribution{T}})

A method that returns the element-type `T` of a distribution type.
"""
function Base.eltype(dist::Type{ <: Distribution{T}}) where {T}
    return T
end

"""
    DistDefault{T <: SketchDirection}

A mutable struct representing a default struct for a distribution.
A common "interface" for structs. That is, at the very least
a distribution saves a vector of weights in `dist`, and whether or not
storage has already been initialized for `dist` in `initialized_storage`.
"""
mutable struct DistDefault{T <: SketchDirection} <: Distribution{T}
    dist::Vector{Float64}
    initialized_storage::Bool
end

"""
    Base.eltype(dist::Distribution{T}) 

Function to return the element-type `T` of Distribution{T}.
"""
function Base.eltype(dist::Distribution{T}) where {T}
    return T
end

"""
    initialize!(
    distribution_type::Distribution{Left}, 
    A::AbstractArray
) 

A method for specifying a distribution over the rows of the matrix
`A`. Calls the common interface method `getDistribution!`.
"""
function initialize!(
    distribution_type::Distribution{Left}, 
    A::AbstractArray
) 
    
    # initialize buffer arrays for distribution if not already 
    if !distribution_type.initialized_storage
        distribution_storage = zeros(size(A)[1])
        distribution_type.dist = distribution_storage 
        distribution_type.initialized_storage = true
    end

    # get distribution
    getDistribution!(distribution_type, A)
end

"""
    initialize!(
    distribution_type::Distribution{Right},
    A::AbstractArray
)

A method for specifying a distribution over the columns of the matrix
`A`. Calls the common interface method `getDistribution!`.
"""
function initialize!(
    distribution_type::Distribution{Right},
    A::AbstractArray
)

    # initialize buffer arrays for distribution if not already
    if !distribution_type.initialized_storage
        distribution_storage = zeros(size(A)[2])
        distribution_type.dist = distribution_storage
        distribution_type.initialized_storage = true
    end

    # get distribution
    getDistribution!(distribution_type, A')
end

"""
    getDistribution!(
        distribution_type::Distribution{<:SketchDirection},
        B::AbstractArray
    )

A common interface responsible for creating a distribution over the rows of the matrix B.
The method modifies the struct distribution_type.

The output of this function will depend on `dist::Distribution`.
"""
function getDistribution!(
    distribution_type::Distribution{<:SketchDirection},
    B::AbstractArray
)

end


###############################
# Distribution Imports
###############################
include("distributions/leverage_scores.jl")