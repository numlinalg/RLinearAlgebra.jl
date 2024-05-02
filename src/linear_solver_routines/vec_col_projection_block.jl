
# This file is pat of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements rsubsolve function
# 3. Exports Type

#using LinearAlgebra

"""
    LinSysVecColBlockProj <: LinSysVecColProjection

A mutable structure that represents a standard block column projection method.

# Aliases
- `BlockCoordinateDescent`

# Fields
- `alpha::Float64`, a relaxation parameter that should be set between `0.0` and `2.0`
- `G::GentData`, a buffer to store important information related to the Gentleman's 
incremental QR least squares solver.
Calling `LinSysVecColBlockProj()` defaults the relaxatoin parameter to `1.0`.

"""
mutable struct LinSysVecColBlockProj <: LinSysVecColProjection
    α::Float64
    G::Union{Nothing, GentData}
    update::Union{Nothing, AbstractArray}
end
LinSysVecColBlockProj(α) = LinSysVecColBlockProj(α, nothing, nothing)

LinSysVecColBlockProj() = LinSysVecColBlockProj(1.0, nothing, nothing)

BlockCoordinateDescent = LinSysVecColBlockProj

# Common rsubsolve interface for linear systems
function rsubsolve!(
    type::LinSysVecColBlockProj,
    x::AbstractVector,
    samp::Tuple{U,V,W,X} where {U<:Union{AbstractVector,AbstractMatrix},V<:AbstractArray,W<:AbstractVector,X<:AbstractVector},
    iter::Int64,
)
    # samp[1] is the search direction
    # samp[2] is the sketched matrix A
    # samp[3] is the residual of system in A * samp[1], samp[1]' A' * (A * x - b)
    # samp[4] is the residual of the system A * x - b
    if iter == 1
        m,p = size(samp[2])
        rowBlockSize = m < 100000 ? m : min(div(m, 10), 100000)
        type.G = Gent(samp[2], min(rowBlockSize, 10000))
        type.update = Array{typeof(samp[2][1])}(undef, p)
    end
    type.G.A = samp[2]
    ldiv!(type.update, type.G, samp[4])
    update_sol!(x, type.update, samp[1], type.α)

    return nothing
end
