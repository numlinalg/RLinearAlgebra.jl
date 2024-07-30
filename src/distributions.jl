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
    Distribution

Abstract supertype for distributions used in sampling and sketching components of a linear system.
"""
abstract type Distribution end

"""
    RowDistribution <: Distribution

Abstract supertype for distributions used in row sampling and row sketching
of a linear system.

# Aliases
- `RowDist`
"""
abstract type RowDistribution <: Distribution end
RowDist = RowDistribution

"""
    ColDistribution <: Distribution

Abstract supertype for distribution used in column sampling and column sketching of a linear system.

# Aliases
- `ColDistribution`
- `ColDist`
"""
abstract type ColumnDistribution <: Distribution end
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

The value returned by `getDistribution` will depend on the subtype `Distribution` being used. 
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
include("distributions/row_dist_frobenius_norm.jl")
include("distributions/row_dist_leverage_score.jl")
include("distributions/row_dist_approximate_leverage_score.jl")


###############################
# Column Distributions
###############################

# Non-adaptive
include("distributions/col_dist_frobenius_norm.jl")
include("distributions/col_dist_leverage_score.jl")
include("distributions/col_dist_approximate_leverage_score.jl")