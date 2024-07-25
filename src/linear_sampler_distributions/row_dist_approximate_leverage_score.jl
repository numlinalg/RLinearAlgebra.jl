# This file is part of RLinearAlgebra.jl
# 1. Specifies a type of distribution 
# 2. Specifies a initialization procedure for the distribution
#
# Date: 07/22/2024
# Author: Christian Varner
# Purpose: Implement a row distribution using approximate leverage scores.

"""
    RowDistApproximateLeverageScores <: RowDistribution

A immutable struct that represents a distribution over the rows using approximated leverage scores. The assumption is that A has full rank.

See Petros Drineas, , Malik Magdon-Ismail, Michael W. Mahoney, David P. Woodruff. 
"Fast approximation of matrix coherence and statistical leverage." (2012).

# Fields

- `Π_1::AbstractArray`, sketch matrix of size (r1, size(A)[1]), where r1 is chosen by the user.
- `Π_2::AbstractArray`, sketch matrix of size (size(A)[2], r2), where r2 is chosen by the user.
"""
struct RowDistApproximateLeverageScores <: RowDistribution 
    Π_1::AbstractArray
    Π_2::AbstractArray
end

# common interface
function getDistribution(
    distribution::RowDistApproximateLeverageScores,
    A::AbstractArray
)

    # compute svd 
    # TODO: problem when sketch size smaller than number of columns
    F = svd( distribution.Π_1 * A; full = true)
    Ω = A * F.Vt' * Diagonal(F.S .^ (-1)) * distribution.Π_2

    # approximated leverage scores
    n = size(Ω)[1]
    dist = zeros(n)
    @inbounds for i in 1:n
        dist[i] = norm(@view Ω[i, :])^2
    end

    # normalize and return
    dist ./= sum(dist)
    return Weights(dist)
end