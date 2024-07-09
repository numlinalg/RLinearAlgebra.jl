# This file is pat of RLinearAlgebra.jl
"""
    LinSysBlkRowLQ <: LinSysBlkRowProjection

A mutable structure that represents a standard block row projection method.

# Aliases
- `BlockKaczmarz`

# Fields
- `α::Float64`, a relaxation parameter that should be set between `0.0` and `2.0`
- `update::Union{Nothing, AbstractArray}`, a buffer for storing update.

# Constructors
Calling `LinSysBlkRowProj()` defaults the relaxation parameter to `1.0`.
"""
mutable struct LinSysBlkRowLQ <: LinSysBlkRowProjection 
    α::Float64
    update::Union{Nothing, Vector{Float64}}
end
LinSysBlkRowLQ(α) = LinSysBlkRowLQ(α, nothing)

LinSysBlkRowLQ() = LinSysBlkRowLQ(1.0, nothing)

BlockKaczmarz = LinSysBlkRowLQ

# Common rsubsolve interface for linear systems
function rsubsolve!(
    type::LinSysBlkRowLQ,
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

    fill!(type.update, 0.0)
    # Solve the underdetemined system 
    LinearAlgebra.ldiv!(type.update, factorize(samp[2]), samp[3])
    x .-= type.α * type.update
    return nothing
end
