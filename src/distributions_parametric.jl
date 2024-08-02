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
## - `initialize` function documentation and interface
## - `getDistribution` function documentation and interface.
## - Include statements
##
##############################################################################

"""
"""
abstract type SketchDirection end 

"""
"""
abstract type Left <: SketchDirection end # comes from rows

"""
"""
abstract type Right <: SketchDirection end # comes from columns

"""
    Distribution{T <: SketchDirection}

Parametric type that represents a distribution over the rows when
`T <: Left`, and over the columns when `T <: Right`. Corresponds
to left and right sketching of a matrix.
"""
abstract type ParametricDistribution{T <: SketchDirection} end

"""
    Base.eltype(dist::Distribution{T}) 

Function to return the element-type `T` of Distribution{T}.
"""
function Base.eltype(dist::ParametricDistribution{T}) where {T}
    return T
end

"""
    initialize(
        dist::Distribution{<:SketchDirection}, 
        A::AbstractArray
    ) 

A method for specifying a distribution over the rows or columns of the matrix
`A`, depending on the element-type parameter `T`. Calls the common interface method
`getDistribution`.

The value returned by `initialize` will depend on the specific `Distribution` that is used
and the parameter `T`.
- For `T <: Left`, a `Weights` vector of size `size(A)[1]` will be returned that has sum approximately 1 using the method specified by `Distribution`.
- For `T <: Right`, a `Weights` vector of size `size(A)[2]` will be returned that has sum approximately 1 using the method specified by `Distribution`.
"""
function initialize(
    dist::ParametricDistribution{<:SketchDirection}, 
    A::AbstractArray
) where {T}

    if T == Left
        B = A
    elseif T == Right
        B = A'
    elseif T == SketchDirection
        B = A
        @warn("SketchDirection is ambigious, returning distribution over rows of A.")
    end
    return getDistributionParametric(dist, B)
end

"""
    getDistribution(
        distribution_type::Distribution{<:SketchDirection},
        B::AbstractArray
    )

A common interface responsible for creating a distribution over the rows of the matrix B.

The output of this function will depend on `dist::Distribution`.
"""
function getDistributionParametric(
    distribution_type::ParametricDistribution{<:SketchDirection},
    B::AbstractArray
)

    return nothing
end


###############################
# Distribution Imports
###############################
include("parametric_distributions/parametric_frobenius_norm.jl")
include("parametric_distributions/parametric_leverage_scores.jl")
include("parametric_distributions/parametric_approximate_leverage_scores.jl")