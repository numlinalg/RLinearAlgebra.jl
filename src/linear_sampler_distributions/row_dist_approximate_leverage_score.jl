# Date: 07/22/2024
# Author: Christian Varner
# Purpose: Implement a row distribution using approximate leverage scores.

"""
    RowDistApproximateLeverageScores <: RowDistribution

A immutable struct that represents a distribution over the rows using approximated
leverage scores.

See Petros Drineas, , Malik Magdon-Ismail, Michael W. Mahoney, David P. Woodruff. 
"Fast approximation of matrix coherence and statistical leverage." (2012).

# Fields

- `Π_1::AbstractArray`, sketch matrix of size (r1, size(A)[1]), where r1 is showen by the user.
- `Π_2::AbstractArray`, sketch matrix of size (size(A)[2], r2), where r2 is chosen by the user.
"""
struct RowDistApproximateLeverageScores <: RowDistribution 
    Π_1::AbstractArray
    Π_2::AbstractArray
end

function getDistribution(
    distribution::RowDistApproximateLeverageScores,
    A::AbstractArray
)

    F = svd(distribution.Π_1*A; full=true) # TODO: do you need to compute the full distribution?
    ARinv = A*F.Vt'*Diagonal(F.S .^ (-1))
    Ω = ARinv*distribution.Π_2

    dist = zeros(size(Ω)[1])
    for i in 1:size(Ω)[1]
        dist[i] = norm(Ω[i,:])^2
    end
    dist /= sum(dist)

    return dist
end