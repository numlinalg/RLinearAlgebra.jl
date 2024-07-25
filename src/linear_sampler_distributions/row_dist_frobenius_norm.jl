# This file is part of RLinearAlgebra.jl
# 1. Specifies a type of distribution 
# 2. Specifies a initialization procedure for the distribution
#
# Date: 07/22/2024
# Author: Christian Varner
# Purpose: Implementation of a distribution over rows using the Frobenius Norm.

"""
    RowDistFrobeniusNorm <: RowDistribution

An immutable struct that represents a row distribution initialized via
the frobenius norm of `A`. More specifically, the probability of selecting row i, is norm(A[i,:])^2/norm(A)^2. Note that norm(A) == ||A||_F.
"""
struct RowDistFrobeniusNorm <: RowDistribution  end

# common interface for distributions
function getDistribution(
    distribution::RowDistFrobeniusNorm,
    A::AbstractArray
)
    # get the norm of the rows of A
    nrow = size(A)[1]
    dist = zeros(nrow)
    @inbounds for i in 1:nrow
        dist[i] = norm(@view A[i,:])^2
    end

    # normalize distribution and return
    dist .= dist ./ (norm(A)^2)
    return Weights(dist)
end