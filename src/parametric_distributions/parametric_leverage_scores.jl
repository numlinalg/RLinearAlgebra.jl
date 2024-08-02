# Date: 08/01/2024
# Author: Christian Varner
# Purpose: Implementation of a distribution using leverage scores

"""
    DistLeverageScore{T <: SketchDirection} <: Distribution{T}

An immutable struct that represents a distribution created by using the leverage scores
over the rows of a matrix `B`.

See Petros Drineas, Malik Magdon-Ismail, Michael W. Mahoney, David P. Woodruff. 
"Fast approximation of matrix coherence and statistical leverage." (2012).

# Additional Constructors
`DistLeverageScore(x::Type{T})`, is `DistLeverageScore{T}()` for `T <: SketchDirection`
`DistLeverageScore(left::Bool)`, is `DistLeverageScore{Left}()` if `left == true`, otherwise `DistLeverageScore{Right}()`
"""
struct DistLeverageScore{T<:SketchDirection} <: ParametricDistribution{T} end

# constructor
DistLeverageScore(x::Type{T}) where T <: SketchDirection = DistLeverageScore{T}()
DistLeverageScore(left::Bool) = left ? DistLeverageScore{Left}() : DistLeverageScore{Right}()

# common interface
function getDistributionParametric(
    distribution_type::DistLeverageScore{<:SketchDirection},
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