# This file is pat of RLinearAlgebra.jl
# This was written by Nathaniel Pritchard
"""
    LinSysBlkColProj <: LinSysBlkColProjection

A mutable structure that represents a standard block column projection method.

# Aliases
- `BlockCoordinateDescent`

# Fields
- `alpha::Float64`, a relaxation parameter that should be set between `0.0` and `2.0`
- `G::GentData`, a buffer to store important information related to the Gentleman's 
incremental QR least squares solver.

Calling the constructor `LinSysBlkColProj()` defaults the relaxation parameter to `1.0`.

"""
mutable struct LinSysBlkColProj <: LinSysBlkColProjection 
    α::Float64
    G::Union{Nothing, GentData}
    update::Union{Nothing, AbstractArray}
end
LinSysBlkColProj(α) = LinSysBlkColProj(α, nothing, nothing)

LinSysBlkColProj() = LinSysBlkColProj(1.0, nothing, nothing)

BlockCoordinateDescent = LinSysBlkColProj

# Common rsubsolve interface for linear systems
function rsubsolve!(
    type::LinSysBlkColProj,
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
        # If there are less than 10000 rows, m < 10000, perform gentlemans with block size m otherwise keep the block size 
        # less than 10000
        rowBlockSize = m < 10000 ? m : min(div(m, 10), 10000)
        # Gentleman's will not use more than 10000 rows as a block 
        type.G = Gent(samp[2], rowBlockSize)
        type.update = Array{typeof(samp[2][1])}(undef, p)
    end
    type.G.A = samp[2]
    ldiv!(type.update, type.G, samp[4])
    update_sol!(x, type.update, samp[1], type.α)

    return nothing
end