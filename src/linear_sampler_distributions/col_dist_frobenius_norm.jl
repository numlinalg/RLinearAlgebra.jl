# This file is part of RLinearAlgebra.jl
# 1. Specifies a type of distribution 
# 2. Specifies a initialization procedure for the distribution
#
# Date: 07/22/2024
# Author: Christian Varner
# Purpose: Implementation of a distribution over columns using the Frobenius Norm.

"""
   ColDistFrobeniusNorm <: ColDistribution

An immutable struct with no fields that represents a column distribution
initialized using the frobenius norm of `A`. More specifically, the probability
of selecting column i, is norm(A[:,i])^2/norm(A)^2. Note that norm(A) == ||A||_F.
"""
struct ColDistFrobeniusNorm <: ColDistribution end

# common interface to get distributions
function getDistribution(
    distribution::ColDistFrobeniusNorm,
    A::AbstractArray
)
    # compute the norm of the columns
    ncol = size(A)[2]
    dist = zeros(ncol)
    @inbounds for i in 1:ncol
        dist[i] = norm(@view A[:,i])^2
    end

    # normalize the distribution and return
    dist .= dist ./ (norm(A)^2)
    return Weights(dist)
end