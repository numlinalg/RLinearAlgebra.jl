# Date: 07/22/2024
# Author: Christian Varner
# Purpose: Implement a distribution using Leverage Scores

"""
"""
struct ColDistLeverageScore end

function getDistribution(
    distribution::ColDistLeverageScore,
    A::AbstractArray
)

    # compute QR decomposition
    Q = Matrix(qr(A').Q)
    nrow = size(Q)[1]
    
    # form distribution
    dist = zeros(nrow)
    for i in 1:nrow
        dist[i] = norm(Q[i,:])
    end
    dist /= sum(dist)

    return Weights(dist)

end