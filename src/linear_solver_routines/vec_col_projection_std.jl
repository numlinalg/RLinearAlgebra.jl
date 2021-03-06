# This file is pat of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements rsubsolve function
# 3. Exports Type

#using LinearAlgebra

"""
    LinSysVecColProjStd <: LinSysVecColProjection

A mutable structure that represents a standard column projection method.

See Luo, Zhi-Quan, and Paul Tseng. "On the convergence of the coordinate descent method
    for convex differentiable minimization." Journal of Optimization Theory and
    Applications 72.1 (1992): 7-35.

For a generalization, see Patel, Vivak, Mohammad Jahangoshahi, and Daniel Adrian Maldonado.
    "An Implicit Representation and Iterative Solution of Randomly Sketched Linear Systems."
    SIAM Journal on Matrix Analysis and Applications (2021) 42:2, 800-831.
    https://doi.org/10.1137/19M1259481

# Aliases
- `CoordinateDescent`
- `GaussSeidel`

# Fields
- `alpha::Float64`, a relaxation parameter that should be set between `0.0` and `2.0`

Calling `LinSysVecColProjStd()` defaults the relaxatoin parameter to `1.0`.

"""
mutable struct LinSysVecColProjStd <: LinSysVecColProjection
    α::Float64
end
CoordinateDescent = LinSysVecColProjStd
GaussSeidel = LinSysVecColProjStd

LinSysVecColProjStd() = LinSysVecColProjStd(1.0)

# Common rsubsolve interface for linear systems
function rsubsolve!(
    type::LinSysVecColProjStd,
    x::AbstractVector,
    samp::Tuple{U,V,W} where {U<:AbstractVector,V<:AbstractArray,W<:Real},
    iter::Int64,
)
    # samp[1] is the search direction
    # samp[2] is the coefficient matrix A
    # samp[3] is the residual of normal system in A * samp[1], samp[1]' A' * (A * x - b)

    x .= x - samp[1] * (type.α * samp[3] / sum((samp[2] * samp[1]).^2))

    return nothing
end
