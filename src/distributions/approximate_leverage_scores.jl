# Date: 08/01/2024
# Author: Christian Varner
# Purpose: Implementing a distribution using approximate leverage scores

"""
    DistApproximateLeverageScore{T <: SketchDirection} <: Distribution{T}

An mutable struct that represents a distribution over the rows using
approximated leverage scores. 

See Petros Drineas, , Malik Magdon-Ismail, Michael W. Mahoney, David P. Woodruff. 
"Fast approximation of matrix coherence and statistical leverage." (2012).

# Fields

- `Π_1::Union{Matrix{Float64}, Matrix{Int64}}`, sketch matrix of size `(r1, size(B)[1])`, where `r1` is chosen by the user.
- `Π_2::Union{Matrix{Float64}, Matrix{Int64}}`, sketch matrix of size `(size(B)[2], r2)`, where `r2` is chosen by the user.
- `dist::Vector{Float64}`, buffer vector that stores the probability vector.
- `initialized_storage::Bool`, if the buffer vector `dist` has been initialized.
"""
mutable struct DistApproximateLeverageScore{T <: SketchDirection} <: Distribution{T} 
    Π_1::Union{Matrix{Float64}, Matrix{Int64}}
    Π_2::Union{Matrix{Float64}, Matrix{Int64}}
    dist::Vector{Float64}
    initialized_storage::Bool
end

# constructors
function DistApproximateLeverageScore(
    sketch_direction::Type{T},
    Π_1::Union{Matrix{Float64}, Matrix{Int64}}, 
    Π_2::Union{Matrix{Float64}, Matrix{Int64}};
    dist = zeros(1),
    flag = false
) where T
    return DistApproximateLeverageScore{T}(Π_1, Π_2, dist, flag)
end

# common interface
function getDistribution!(
    distribution_type::DistApproximateLeverageScore{<:SketchDirection},
    B::AbstractArray
)

    # error checking
    if size(distribution_type.Π_1)[1] < size(B)[2]
        throw(BoundsError("The dimension of the left sketching matrix Pi_1 is too small!"))
    elseif size(B)[1] <= size(B)[2]
        throw(DomainError("Number of observations is the same as parameters in A!"))
    end

    # compute svd  
    sketched_B = distribution_type.Π_1 * B
    F = svd( sketched_B; full = true)
    Ω = B * F.Vt' * Diagonal(F.S .^ (-1)) * distribution_type.Π_2

    # approximated leverage scores
    for i in 1:size(distribution_type.dist)[1]
        distribution_type.dist[i] = norm( @view Ω[i, :] )^2 # why is this taking so many allocation -> Nathaniel
    end

    # normalize and return
    distribution_type.dist ./= sum(distribution_type.dist)
end
