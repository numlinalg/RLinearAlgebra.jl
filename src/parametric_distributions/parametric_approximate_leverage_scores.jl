# Date: 08/01/2024
# Author: Christian Varner
# Purpose: Implementing a distribution using approximate leverage scores

struct DistApproximateLeverageScore{T<:SketchDirection} <: ParametricDistribution{T} 
    Π_1::Union{Matrix{Float64}, Matrix{Int64}}
    Π_2::Union{Matrix{Float64}, Matrix{Int64}}
end

# common interface
function getDistributionParametric(
    distribution::DistApproximateLeverageScore{<:SketchDirection},
    B::AbstractArray
)

    # compute svd 
    # TODO: problem when sketch size smaller than number of columns since F.S can be at most r1 < size(A)[2]
    sketched_B = distribution.Π_1 * B
    F = svd( sketched_B; full = true)
    Ω = B * F.Vt' * Diagonal(F.S .^ (-1)) * distribution.Π_2

    # approximated leverage scores
    dist = zeros(size(B)[1])
    @inbounds for i in 1:size(dist)[1]
        dist[i] = norm( @view Ω[i, :] )^2 # why is this taking so many allocation -> Nathaniel
    end

    # normalize and return
    dist ./= sum(dist)
    return Weights(dist)
end