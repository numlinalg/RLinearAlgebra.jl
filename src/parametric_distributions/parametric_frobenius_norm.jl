# Date: 08/01/2024
# Author: Christian Varner
# Purpose: Implementing a parametric type and function
# for a distribution based on the frobenius norm.

"""
   DistFrobeniusNorm{T <: SketchDirection} <: Distribution{T}

An immutable struct that represents creating a distribution based on the row norms
of a matrix `B`. Specifically, the probability of row `i` is `norm(B[i, :])^2/norm(B)^2`.

# Additional Constructors
`DistFrobeniusNorm(x::Type{T})`, is `DistFrobeniusNorm{T}()` where `T <: SketchDirection`.
`DistFrobeniusNorm(left::Bool)`, where if `left == true` `DistFrobeniusNorm{Left}()` is called, otherwise `DistFrobeniusNorm{Right}()`.
"""
struct DistFrobeniusNorm{T <: SketchDirection} <: ParametricDistribution{T} end 

# constructors
DistFrobeniusNorm(x::Type{T}) where T <: SketchDirection = DistFrobeniusNorm{T}()
DistFrobeniusNorm(left::Bool) = left ? DistFrobeniusNorm{Left}() : DistFrobeniusNorm{Right}()

# common interface
function getDistributionParametric(
    distribution_type::DistFrobeniusNorm{<:SketchDirection},
    B::AbstractArray
)

    # get the norm of the rows of B
    dim = size(B)[1]
    dist = zeros(dim)
    @inbounds for i in 1:dim
        dist[i] = norm(@view B[i, :])^2
    end

    # normalize distribution and return
    dist .= dist ./ (norm(B)^2)
    return Weights(dist)
end