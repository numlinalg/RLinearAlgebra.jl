# Date: 07/22/2024
# Author: Christian Varner
# Purpose: Implement a row distribution using approximate leverage scores.

"""
"""
struct RowDistApproximateLeverageScores 
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