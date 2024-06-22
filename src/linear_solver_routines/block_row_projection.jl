# This file is pat of RLinearAlgebra.jl
"""
    LinSysBlkRowProj <: LinSysBlkRowProjection

A mutable structure that represents a standard block row projection method.

# Aliases
- `BlockKaczmarz`

# Fields
- `alpha::Float64`, a relaxation parameter that should be set between `0.0` and `2.0`
- `update::Union{Nothing, AbstractArray}` - A buffer for storing update.

Calling `LinSysBlkRowProj()` defaults the relaxatoin parameter to `1.0`.
"""
mutable struct LinSysBlkRowProj <: LinSysBlkRowProjection 
    α::Float64
    update::Union{Nothing, AbstractArray}
end
LinSysBlkRowProj(α) = LinSysBlkRowProj(α, nothing)

LinSysBlkRowProj() = LinSysBlkRowProj(1.0, nothing)

BlockKaczmarz = LinSysBlkRowProj
# Common rsubsolve interface for linear systems
function rsubsolve!(
    type::LinSysBlkRowProj,
    x::AbstractVector,
    samp::Tuple{U,V,W} where {U<:Union{AbstractVector,AbstractMatrix},V<:AbstractArray,W<:AbstractVector},
    iter::Int64,
)
    # samp[1] is the search direction
    # samp[2] is the sketched matrix A
    # samp[3] is the residual of system in A * samp[1], (samp[1] * A * x - b)
    if iter == 1
        # get the dimensions to allocate the update vector
        p,n = size(samp[2])
        type.update = Array{typeof(samp[2][1])}(undef, n)
    end
    # Reset the update vector 
    fill!(type.update, 0)
    ldiv!(type.update, lq(samp[2]), samp[3])
    x .-= type.alpha .* type.update
    return nothing
end
