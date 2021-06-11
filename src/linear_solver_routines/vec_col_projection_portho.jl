# 1. Specifies type
# 2. Implements rsubsolve function
# 3. Exports Type

#using LinearAlgebra

"""
    LinSysVecColProjPO <: LinSysVecColProjection

A mutable structure that represents the standard column projection method
    with orthogonalization of the current projection against a set of `m` previous
    projection directions, where `m` is specified by the user.

See Patel, Vivak, Mohammad Jahangoshahi, and Daniel Adrian Maldonado. "An Implicit
    Representation and Iterative Solution of Randomly Sketched Linear Systems."
    SIAM Journal on Matrix Analysis and Applications (2021) 42:2, 800-831.
    https://doi.org/10.1137/19M1259481

# Fields

- `α::Float64`, the relaxation parameter (usually between `0.0` and `2.0`)
- `m::Int64`, the number of previous directions against which to orthogonalize
- `Z::Union{Vector{Vector{Float64}}, Nothing}`, stores the `m` vectors against which the
    current projection is orthogonalized against

Calling `LinSysVecColProjPO()` defaults the relaxation parameter to `1.0`, the memory
    parameter `m` to `5`, and `Z` to `nothing`.
"""
mutable struct LinSysVecColProjPO <: LinSysVecColProjection
    α::Float64
    m::Int64
    Z::Union{Vector{Vector{Float64}}, Nothing}
end
LinSysVecColProjPO() = LinSysVecColProjPO(1.0, 5, nothing)

# Common rsubsolve interface for linear systems
function rsubsolve!(
    type::LinSysVecColProjPO,
    x::Vector{Float64},
    samp::Tuple{U,V,W} where {U<:Vector,V<:Matrix,W<:Real},
    iter::Int64,
)
    # samp[1] is the search direction
    # samp[2] is the coefficient matrix, A
    # samp[3] is a projection of the residual of normal system, samp[1]' * A' * (A * x - b)

    # Allocate space for orthogonalization set Z
    if iter == 1
        d = length(x)
        type.Z = Vector{Float64}[zeros(Float64, d) for i in 1:type.m]
    end

    # Compute orthogonal complement of A' * A * samp[1] using modified Gram-Schmidt
    u = samp[2]' * (samp[2] * samp[1])
    for z in type.Z
        u = u - dot(z, u) * z
    end

    # Check whether vector u is nearly zero
    nrmUsq = dot(u, u)
    if nrmUsq < 1e-15 * length(x); return nothing; end

    # Otherwise, update iterate
    x .= x - u * (type.α * samp[3] / nrmUsq )

    # Update orthonormal set
    z = u / sqrt(nrmUsq)
    type.Z .= push!(type.Z[2:end], z)

    return nothing
end
