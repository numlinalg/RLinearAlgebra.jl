# Date: 08/01/2024
# Author: Christian Varner
# Purpose: Implementation of a distribution using leverage scores

"""
    DistLeverageScore{T <: SketchDirection} <: Distribution{T}

An mutable struct that represents a distribution created by using the leverage scores
over the rows of a matrix `B`.

See Petros Drineas, Malik Magdon-Ismail, Michael W. Mahoney, David P. Woodruff. 
"Fast approximation of matrix coherence and statistical leverage." (2012).

# Fields

- `dist::Vector{Float64}`, buffer vector that stores the probability vector.
- `initialized_storage::Bool`, if the buffer vector `dist` has been initialized.
"""
mutable struct DistLeverageScore{T<:SketchDirection} <: Distribution{T} 
    dist::Vector{Float64}
    initialized_storage::Bool
end

# constructor
function DistLeverageScore(x::Type{T}; dist = zeros(1), flag = false) where T
    return DistLeverageScore{T}(dist, flag)
end

# common interface
function getDistribution!(
    distribution_type::DistLeverageScore{<:SketchDirection},
    B::AbstractArray
)

    # compute QR decomposition
    Q1 = Matrix(qr(B).Q) # get thin Q
    dim = size(Q1)[1]
    
    # compute the leverage scores
    @inbounds for i in 1:dim
        distribution_type.dist[i] = norm(@view Q1[i, :])^2
    end

    # normalize and return
    distribution_type.dist ./= sum(distribution_type.dist)
end