# This file is part of RLinearAlgebra.jl
# 1. Specifies a type of distribution 
# 2. Specifies a initialization procedure for the distribution
#
# Date: 07/22/2024
# Author: Christian Varner
# Purpose: Implement a distribution over rows using Leverage Scores

"""
    RowDistLeverageScore <: RowDistribution

An immutable struct that represents a distribution over the rows of
`A` that is initialized using the leverage scores.
"""
struct RowDistLeverageScore <: RowDistribution end

# common interface
function getDistribution(
    distribution::RowDistLeverageScore,
    A::AbstractArray
)

    # compute QR decomposition
    Q1 = Matrix(qr(A).Q) # get thin Q
    nrow = size(Q1)[1]
    
    # compute the leverage scores
    dist = zeros(nrow)
    @inbounds for i in 1:nrow
        dist[i] = norm(@view Q1[i, :])^2
    end

    # normalize and return
    dist ./= sum(dist)
    return Weights(dist)
end