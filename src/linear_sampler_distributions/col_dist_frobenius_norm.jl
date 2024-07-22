# This file is part of RLinearAlgebra.jl
# 1. Specifies a type of distribution 
# 2. Specifies a initialization procedure for the distribution
#
# Date: 07/22/2024
# Author: Christian Varner
# Purpose: Implementation of a distribution over col using the Frobenius Norm.

"""
"""
struct ColDistFrobeniusNorm end

function getDistribution(
    distribution::ColDistFrobeniusNorm,
    A::AbstractArray
)
    ncol = size(A)[2]
    distribution = zeros(ncol)
    for i in 1:ncol
        distribution[i] = norm(A[:,i])^2
    end
    distribution .= distribution ./ (norm(A)^2)

    return Weights(distribution)
end