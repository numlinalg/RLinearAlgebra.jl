# This file is pat of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements rsubsolve function
# 3. Exports Type

#using LinearAlgebra

"""
    LinSysVecRowProjFO <: LinSysVecRowProjection

A mutable structure that represents the standard row projection method
with full orthogonalization against all previous projections. Equivalently, this type
represents a solver for incrementally constructed matrix sketches.

See McCormick, S. F. "The methods of Kaczmarz and row orthogonalization for solving linear
    equations and least squares problems in Hilbert space." Indiana University Mathematics
    Journal 26.6 (1977): 1137-1150. https://www.jstor.org/stable/24891603

See Patel, Vivak, Mohammad Jahangoshahi, and Daniel Adrian Maldonado. "An Implicit
    Representation and Iterative Solution of Randomly Sketched Linear Systems."
    SIAM Journal on Matrix Analysis and Applications (2021) 42:2, 800-831.
    https://doi.org/10.1137/19M1259481

# Fields

- `S::Union{Matrix{Float64}, Nothing}` is a matrix used for orthogonalizing against
    all previous search directions.

Calling `LinSysVecRowProjFO()` defaults to `LinSysVecRowProjFO(nothing)`.
"""
mutable struct LinSysVecRowProjFO <: LinSysVecRowProjection
    S::Union{Matrix{Float64}, Nothing}
end
LinSysVecRowProjFO() = LinSysVecRowProjFO(nothing)

# Common rsubsolve interface for linear systems
function rsubsolve!(
    type::LinSysVecRowProjFO,
    x::AbstractVector,
    samp::Tuple{U,V} where {U<:AbstractVector,V<:Real},
    iter::Int64,
)
    # samp[1] is vector in the row space of the coefficient matrix
    # samp[2] is a scalar corresponding to samp[1]

    # Allocate space for orthogonalization matrix S
    if iter == 1
        d = length(x)
        type.S = diagm(ones(Float64, d))
    end

    #Compute orthogonal component of q using S (projection matrix)
    u = type.S * samp[1]

    if dot(u,u) < 1e-30 * length(x)
        return nothing
    end

    #Update Iterate
    res = samp[2] - dot(samp[1], x)
    γ = dot(u, samp[1])
    x .= x + u * ( res / γ )

    #Update Projection matrix
    type.S .= (I - (u/γ)*samp[1]')*type.S

    return nothing
end
