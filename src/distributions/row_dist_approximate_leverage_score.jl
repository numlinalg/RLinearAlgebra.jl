# This file is part of RLinearAlgebra.jl
# 1. Specifies a type of distribution 
# 2. Specifies a initialization procedure for the distribution
#
# Date: 07/22/2024
# Author: Christian Varner
# Purpose: Implement a row distribution using approximate leverage scores.

"""
    RowDistApproximateLeverageScores <: RowDistribution

An immutable struct that represents a distribution over the rows using approximated leverage scores. 
The assumption is that A has full rank.

See Petros Drineas, Malik Magdon-Ismail, Michael W. Mahoney, David P. Woodruff. 
"Fast approximation of matrix coherence and statistical leverage." (2012).

# Fields

- `Π_1::Union{Matrix{Float64}, Matrix{Int64}}`, sketch matrix of size (r1, size(A)[1]), where r1 is chosen by the user.
- `Π_2::Union{Matrix{Float64}, Matrix{Int64}}`, sketch matrix of size (size(A)[2], r2), where r2 is chosen by the user.
"""
struct RowDistApproximateLeverageScores <: RowDistribution 
    Π_1::Union{Matrix{Float64}, Matrix{Int64}}
    Π_2::Union{Matrix{Float64}, Matrix{Int64}}
end

# common interface
function getDistribution(
    distribution::RowDistApproximateLeverageScores,
    A::AbstractArray
)

    # compute svd 
    # TODO: problem when sketch size smaller than number of columns since F.S can be at most r1 < size(A)[2]
    sketched_A = distribution.Π_1 * A 
    F = svd( sketched_A; full = true)
    Ω = A * F.Vt' * Diagonal(F.S .^ (-1)) * distribution.Π_2

    # approximated leverage scores
    dist = zeros(size(A)[1])
    @inbounds for i in 1:size(dist)[1]
        dist[i] = norm( @view Ω[i, :] )^2 # why is this taking so many allocation -> Nathaniel
    end

    # normalize and return
    dist ./= sum(dist)
    return Weights(dist)
end