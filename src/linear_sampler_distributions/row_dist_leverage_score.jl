# Date: 07/22/2024
# Author: Christian Varner
# Purpose: Implement a distribution using Leverage Scores

"""
"""
struct RowDistLeverageScore end

function getDistribution(
    distribution::RowDistLeverageScore,
    A::AbstractArray
)

    # compute QR decomposition
    Q = Matrix(qr(A).Q)
    nrow = size(Q)[1]
    
    # form distribution
    distribution = zeros(size(A)[1])
    d = 0
    for i in 1:nrow
        d += norm(Q[i,:])
        distribution[i] = norm(A[i,:])
    end
    distribution /= d

    return Weights(distribution)

end