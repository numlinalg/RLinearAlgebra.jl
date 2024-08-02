# Date: 08/01/2024
# Author: Christian Varner
# Purpose: Implementing a distribution using approximate leverage scores

"""
    DistApproximateLeverageScore{T <: SketchDirection} <: Distribution{T}

An immutable struct that represents a distribution over the rows using
approximated leverage scores. 

See Petros Drineas, , Malik Magdon-Ismail, Michael W. Mahoney, David P. Woodruff. 
"Fast approximation of matrix coherence and statistical leverage." (2012).

# Fields

- `Π_1::Union{Matrix{Float64}, Matrix{Int64}}`, sketch matrix of size `(r1, size(B)[1])`, where `r1` is chosen by the user.
- `Π_2::Union{Matrix{Float64}, Matrix{Int64}}`, sketch matrix of size `(size(B)[2], r2)`, where `r2` is chosen by the user.
"""
struct DistApproximateLeverageScore{T <: SketchDirection} <: ParametricDistribution{T} 
    Π_1::Union{Matrix{Float64}, Matrix{Int64}}
    Π_2::Union{Matrix{Float64}, Matrix{Int64}}
end

# common interface
function getDistributionParametric(
    distribution_type::DistApproximateLeverageScore{<:SketchDirection},
    B::AbstractArray
)

    # compute svd 
    # TODO: problem when sketch size smaller than number of columns since F.S can be at most r1 < size(A)[2]
    sketched_B = distribution_type.Π_1 * B
    F = svd( sketched_B; full = true)
    Ω = B * F.Vt' * Diagonal(F.S .^ (-1)) * distribution_type.Π_2

    # approximated leverage scores
    dist = zeros(size(B)[1])
    @inbounds for i in 1:size(dist)[1]
        dist[i] = norm( @view Ω[i, :] )^2 # why is this taking so many allocation -> Nathaniel
    end

    # normalize and return
    dist ./= sum(dist)
    return Weights(dist)
end