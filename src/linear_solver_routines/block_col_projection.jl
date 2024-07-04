# This file is pat of RLinearAlgebra.jl
"""
    LinSysBlkColGent <: LinSysBlkColProjection

A mutable structure that represents a standard block column projection method.

# Aliases
- `BlockCoordinateDescent`

# Fields
- `alpha::Float64`, a relaxation parameter that should be set between `0.0` and `2.0`
- `gent::GentData`, a buffer to store important information related to the Gentleman's 
incremental QR least squares solver.
- `update::Union{Nothing, AbstractArray}`, a buffer for storing update.
- `rowsize::Int64`, the maximum size of the Gentleman's row blocks. By defaul this is set to 10,000.
Calling the constructor `LinSysBlkColProj()` defaults the relaxation parameter to `1.0`.
"""
mutable struct LinSysBlkColGent <: LinSysBlkColProjection 
    α::Float64
    gent::Union{Nothing, GentData}
    update::Union{Nothing, AbstractArray}
    rowsize::Int64
end

LinSysBlkColGent(;α = 1.0, rowsize = 10000) = LinSysBlkColGent(α, nothing, nothing, rowsize)
BlockCoordinateDescent = LinSysBlkColGent

# Common rsubsolve interface for linear systems
function rsubsolve!(
    type::LinSysBlkColGent,
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
        brow_size = m < type.rowsize ? m : min(div(m, 10), type.rowsize)
        # Gentleman's will not use more than 10000 rows as a block 
        type.gent = GentData(samp[2], brow_size)
        type.update = Array{typeof(samp[2][1])}(undef, p)
    end
    type.gent.A = samp[2]
    LinearAlgebra.ldiv!(type.update, type.gent, samp[4])
    x .-= type.α * samp[1] * type.update

    return nothing
end
