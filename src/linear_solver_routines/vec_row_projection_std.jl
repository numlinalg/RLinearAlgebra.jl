# This file is pat of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements rsubsolve function
# 3. Exports Type

#using LinearAlgebra

"""
    LinSysVecRowProjStd <: LinSysVecRowProjection

A mutable structure that represents a standard row projection method.

# Aliases
- `Kaczmarz`, see Kaczmarz, M. S. "Approximate solution of systems of linear equations."
    (1937). Original article is in German.
- `ART`, see Gordon, Richard, Robert Bender, and Gabor T. Herman. "Algebraic reconstruction
    techniques (ART) for three-dimensional electron microscopy and X-ray photography."
    Journal of theoretical Biology 29.3 (1970): 471-481.

# Fields
- `alpha::Float64`, a relaxation parameter that should be set between `0.0` and `2.0`

Calling `LinSysVecRowProjStd()` defaults the relaxation parameter to `1.0`.
"""
mutable struct LinSysVecRowProjStd <: LinSysVecRowProjection
    α::Float64
end
Kaczmarz = LinSysVecRowProjStd
ART = LinSysVecRowProjStd

LinSysVecRowProjStd() = LinSysVecRowProjStd(1.0)

# Common rsubsolve interface for linear systems
function rsubsolve!(
    type::LinSysVecRowProjStd,
    x::AbstractVector,
    samp::Tuple{U,V} where {U<:AbstractVector,V<:Real},
    iter::Int64,
)
    # samp[1] is vector in the row space of the coefficient matrix
    # samp[2] is a scalar corresponding to samp[1]
    res = samp[2] - dot(samp[1], x)
    x .= x + samp[1] * (type.α * res / sum(samp[1].^2))

    return nothing
end

# Export LinSysVecRowProjStd, Kaczmarz, ART
