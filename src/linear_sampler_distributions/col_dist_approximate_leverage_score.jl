# This file is part of RLinearAlgebra.jl
# 1. Specifies a type of distribution 
# 2. Specifies a initialization procedure for the distribution
#
# Date: 07/23/2024
# Author: Christian Varner
# Purpose: Implement a column distribution using approximate leverage scores.

"""
    ColDistApproximateLeverageScores <: ColDistribution

An immutable struct that represents a distribution over the columns using
approximated leverage scores. The assumption is that A' has full rank.

This code is equivalent to src/linear_sampler_distributions/row_dist_approximate_leverage_score.jl
except we apply the method to A'

See Petros Drineas, , Malik Magdon-Ismail, Michael W. Mahoney, David P. Woodruff. 
"Fast approximation of matrix coherence and statistical leverage." (2012).

# Fields

- `Π_1::AbstractArray`, sketch matrix of size (r1, size(A)[2]), where r1 is chosen by the user.
- `Π_2::AbstractArray`, sketch matrix of size (size(A)[1], r2), where r2 is chosen by the user.
"""
struct ColDistApproximateLeverageScores <: ColDistribution
    Π_1::AbstractArray
    Π_2::AbstractArray
end

function getDistribution(
    distribution::ColDistApproximateLeverageScores,
    A::AbstractArray
)

    # compute svd
    F = svd(distribution.Π_1 * A'; full=true) # TODO: do you need to compute the full distribution?
    Ω = A' * F.Vt' * Diagonal(F.S .^ (-1)) * distribution.Π_2

    # approximated leverage scores
    n = size(Ω)[1]
    dist = zeros(n)
    @inbounds for i in 1:n
        dist[i] = norm(@view Ω[i, :])^2
    end

    # normalize and return
    dist ./= sum(dist)
    return dist
end