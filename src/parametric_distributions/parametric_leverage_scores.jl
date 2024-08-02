# Date: 08/01/2024
# Author: Christian Varner
# Purpose: Implementation of a distribution using leverage scores

"""
"""
struct DistLeverageScore{T<:SketchDirection} <: ParametricDistribution{T} end

# constructor
DistLeverageScore(x::Type{T}) where T<:SketchDirection = DistLeverageScore{T}()

# common interface
function getDistributionParametric(
    distribution::DistLeverageScore{<:SketchDirection},
    B::AbstractArray
)

    # compute QR decomposition
    Q1 = Matrix(qr(B).Q) # get thin Q
    dim = size(Q1)[1]
    
    # compute the leverage scores
    dist = zeros(dim)
    @inbounds for i in 1:dim
        dist[i] = norm(@view Q1[i, :])^2
    end

    # normalize and return
    dist ./= sum(dist)
    return Weights(dist)
end