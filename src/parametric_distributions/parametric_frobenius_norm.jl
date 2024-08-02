# Date: 08/01/2024
# Author: Christian Varner
# Purpose: Implementing a parametric type and function
# for a distribution based on the frobenius norm.

"""
"""
struct DistFrobeniusNorm{T<:SketchDirection} <: ParametricDistribution{T} end 

# constructor
DistFrobeniusNorm(x::Type{T}) where T<:SketchDirection = DistFrobeniusNorm{T}()

function getDistributionParametric(
    distribution_type::DistFrobeniusNorm{<:SketchDirection},
    B::AbstractArray
)

    # get the norm of the rows of A
    dim = size(B)[1]
    dist = zeros(dim)
    @inbounds for i in 1:dim
        dist[i] = norm(@view B[i, :])^2
    end

    # normalize distribution and return
    dist .= dist ./ (norm(B)^2)
    return Weights(dist)
end