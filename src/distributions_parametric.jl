# Date: 08/01/2024
# Author: Christian Varner
# Purpose: Parametric types for distributions

##############################################################################
## This file is part of RLinearAlgebra.jl
##
## Overview: abstractions and interface methods for creating a distribution 
## over rows and columns of a matrix. 
##
## Contents
## - Abstract Types
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
"""
abstract type ParametricDistribution{T<:SketchDirection} end
function Base.eltype(dist::ParametricDistribution{T}) where {T}
    return T
end

"""
"""
function getDistributionParametric(
    dist::ParametricDistribution{T},
    A::AbstractArray
) where {T}
    return nothing
end


###############################
# Distributions
###############################
include("parametric_distributions/parametric_frobenius_norm.jl")
include("parametric_distributions/parametric_leverage_scores.jl")
include("parametric_distributions/parametric_approximate_leverage_scores.jl")