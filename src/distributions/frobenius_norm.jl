# Date: 08/01/2024
# Author: Christian Varner
# Purpose: Implementing a parametric type and function
# for a distribution based on the frobenius norm.

"""
   DistFrobeniusNorm{T <: SketchDirection} <: Distribution{T}

An mutable struct that represents creating a distribution based on the row norms
of a matrix `B`. Specifically, the probability of row `i` is `norm(B[i, :])^2/norm(B)^2`.

# Fields

- `dist::Vector{Float64}`, buffer vector that stores the probability vector.
- `initialized_storage::Bool`, if the buffer vector `dist` has been initialized.
"""
mutable struct DistFrobeniusNorm{T <: SketchDirection} <: Distribution{T} 
    dist::Vector{Float64}
    initialized_storage::Bool
end 

# constructors
function DistFrobeniusNorm(x::Type{T}; dist = zeros(1), flag = false) where T 
    return DistFrobeniusNorm{T}(dist, flag)
end

# common interface
function getDistribution!(
    distribution_type::DistFrobeniusNorm{<:SketchDirection},
    B::AbstractArray
)

    # get the norm of the rows of B
    dim = size(B)[1]
    for i in 1:dim
        distribution_type.dist[i] = norm(@view B[i, :])^2
    end

    # normalize distribution and return
    distribution_type.dist .= distribution_type.dist ./ (norm(B)^2)
end
