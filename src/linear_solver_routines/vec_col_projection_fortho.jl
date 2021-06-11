# This file is pat of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements rsubsolve function
# 3. Exports Type

#suing LinearAlgebra

"""
    LinSysVecColProjFO <: LinSysVecColProjection

A mutable structure that represents the standard column projection method
with full orthogonalization against all previous projections.

See Patel, Vivak, Mohammad Jahangoshahi, and Daniel Adrian Maldonado. "An Implicit
    Representation and Iterative Solution of Randomly Sketched Linear Systems."
    SIAM Journal on Matrix Analysis and Applications (2021) 42:2, 800-831.
    https://doi.org/10.1137/19M1259481

# Fields

- `S::Union{Matrix{Float64}, Nothing}` is a matrix used for orthogonalizing against
    all previous search directions.

Calling `LinSysVecColProjFO()` defaults to `LinSysVecColProjFO(nothing)`.
"""
mutable struct LinSysVecColProjFO <: LinSysVecColProjection
    S::Union{Matrix{Float64}, Nothing}
end
LinSysVecColProjFO() = LinSysVecColProjFO(nothing)

# Common rsubsolve interface for linear systems
function rsubsolve!(
    type::LinSysVecColProjFO,
    x::Vector{Float64},
    samp::Tuple{U,V,W} where {U<:Vector,V<:Matrix,W<:Real},
    iter::Int64,
)
    # samp[1] is the search direction
    # samp[2] is the coefficient matrix A
    # samp[3] is the residual of normal system in A * samp[1], samp[1]' A' * (A * x - b)

    if iter == 1
        d = length(x)
        type.S = diagm(ones(Float64, d))
    end

    # Compute orthogonal component of A * samp[1] using S (projection matrix)
    q = samp[2]' * (samp[2] * samp[1])
    u = type.S * q

    if dot(u,u) < 1e-26 * length(x)
        return nothing
    end

    # Update Iterate
    γ = dot(u, q)
    x .= x - u * (samp[3] / γ)

    # Update Projection Matrix
    type.S .= (I - ( u / γ) * q') * type.S

    return nothing
end
