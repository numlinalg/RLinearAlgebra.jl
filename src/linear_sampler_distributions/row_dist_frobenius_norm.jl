# This file is part of RLinearAlgebra.jl
# 1. Specifies a type of distribution 
# 2. Specifies a initialization procedure for the distribution
#
# Date: 07/22/2024
# Author: Christian Varner
# Purpose: Implementation of a distribution over rows using the Frobenius Norm.

"""
"""
struct RowDistFrobeniusNorm end

function getDistribution(
    distribution::RowDistFrobeniusNorm,
    A::AbstractArray
)
    nrow = size(A)[1]
    distribution = zeros(nrow)
    for i in 1:nrow
        distribution[i] = norm(A[i,:])^2
    end
    distribution .= distribution ./ (norm(A)^2)

    return Weights(distribution)
end