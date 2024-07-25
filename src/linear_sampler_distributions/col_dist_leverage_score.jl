# This file is part of RLinearAlgebra.jl
# 1. Specifies a type of distribution 
# 2. Specifies a initialization procedure for the distribution
#
# Date: 07/22/2024
# Author: Christian Varner
# Purpose: Implement a distribution over columns using Leverage Scores

"""
    ColDistLeverageScore <: ColDistribution

An immutable struct that represents a distribution over the columns of
`A` that is initialized using the leverage scores.
"""
struct ColDistLeverageScore <: ColDistribution end

# common interface
function getDistribution(
    distribution::ColDistLeverageScore,
    A::AbstractArray
)

    # compute QR decomposition of A'
    Q1 = Matrix(qr(A').Q) # get thin Q
    nrow = size(A')[1]
    
    # compute the leverage scores
    dist = zeros(nrow)
    @inbounds for i in 1:nrow
        dist[i] = norm(@view Q1[i, :])^2
    end

    # normalize and return
    dist ./= sum(dist)
    return Weights(dist)
end