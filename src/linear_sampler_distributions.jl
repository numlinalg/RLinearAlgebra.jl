##############################################################################
## This file is part of RLinearAlgebra.jl
##
## Overview: abstractions and interface methods for creating a distribution 
## over rows and columns of a matrix, and update a distribution.
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
"""
abstract type LinSysSamplerDistribution end

"""
"""
abstract type RowDistribution <: LinSysSamplerDistribution end

"""
"""
abstract type ColDistribution <: LinSysSamplerDistribution end

function getDistribution(
    distribution::Nothing,
    A::AbstractArray
)
    return nothing
end

###############################
# Row Distributions
###############################


###############################
# Column Distributions
###############################