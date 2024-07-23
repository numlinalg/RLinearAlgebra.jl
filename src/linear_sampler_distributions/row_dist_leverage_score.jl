# Date: 07/22/2024
# Author: Christian Varner
# Purpose: Implement a distribution using Leverage Scores

"""
    RowDistLeverageScore <: RowDistribution

An immutable struct that represents a distribution over the rows of
`A` that is initialized using the leverage scores.
"""
struct RowDistLeverageScore <: RowDistribution end

function getDistribution(
    distribution::RowDistLeverageScore,
    A::AbstractArray
)

    # compute QR decomposition
    Q1 = Matrix(qr(A).Q)
    nrow = size(Q1)[1]
    
    # compute the leverage scores
    dist = zeros(nrow)
    for i in 1:nrow
        dist[i] = norm(Q1[i,:])^2
    end

    # normalize and return
    dist /= sum(dist)
    return Weights(dist)
end