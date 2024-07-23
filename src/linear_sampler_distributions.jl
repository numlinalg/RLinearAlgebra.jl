##############################################################################
## This file is part of RLinearAlgebra.jl
##
## Overview: abstractions and interface methods for creating a distribution 
## over rows and columns of a matrix. 
##
## Contents
## - Abstract Types
## - `getDistribution` function documentation and interface.
## - Row distribution
## - Column distribution
## - Include statements
##
##############################################################################

"""
    LinSysSamplerDistribution

Abstract supertype for distributions used in sampling and sketching components of a linear system.

# Aliases
- `LinSysSketchDistribution`
"""
abstract type LinSysSamplerDistribution end
LinSysSketchDistribution = LinSysSamplerDistribution

"""
    RowDistribution <: LinSysSamplerDistribution

Abstract supertype for distributions used in row sampling and row sketching
of a linear system.

# Aliases
- `RowDist`
"""
abstract type RowDistribution <: LinSysSamplerDistribution end
RowDist = RowDistribution

"""
    ColDistribution <: LinSysSamplerDistribution

Abstract supertype for distribution used in column sampling and column sketching of a linear system.

# Aliases
- `ColDistribution`
- `ColDist`
"""
abstract type ColumnDistribution <: LinSysSamplerDistribution end
ColDistribution = ColumnDistribution
ColDist = ColumnDistribution

##########################################
# `getDistribution` function documentation
##########################################
"""
    getDistribution(distribution <: LinSysSamplerDistributions,
                    A::AbstractArray)

A common interface for specifying a distribution over the rows or columns
of a matrix `A` that is part of a linear system. The argument `distribution` is
used to specify a strategy for initializing a distribution.

The value returned by `getDistribution` will depend on the subtype `LinSysSamplerDistribution` being used. 
- For `T<:RowDistribution`, a `Weights` vector of size `size(A)[1]` is returned that represents a distribution over rows.
- For `T<:ColDistribution`, a `Weights` vector of size `size(A)[2]` is returned that represents a distribution over columns. 
"""
function getDistribution(
    distribution::Nothing,
    A::AbstractArray
)
    return nothing
end

###############################
# Row Distributions
###############################

# Non-adaptive
include("linear_sampler_distributions/row_dist_frobenius_norm.jl")
include("linear_sampler_distributions/row_dist_leverage_score.jl")
include("linear_sampler_distributions/row_dist_approximate_leverage_score.jl")


###############################
# Column Distributions
###############################

# Non-adaptive
include("linear_sampler_distributions/col_dist_frobenius_norm.jl")
include("linear_sampler_distributions/col_dist_leverage_score.jl")
include("linear_sampler_distributions/col_dist_approximate_leverage_score.jl")