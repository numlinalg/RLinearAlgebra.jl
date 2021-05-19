using LinearAlgebra

abstract type RPMProjectionType end

mutable struct ProjectionStdCore <: RPMProjectionType
    α::Float64
end
ProjectionStdCore() = ProjectionStdCore(1.0)

mutable struct ProjectionLowCore <: RPMProjectionType
    α::Float64
    m::Int64
    Z::Union{Vector{Vector{Float64}}, Nothing}
end
ProjectionLowCore() = ProjectionLowCore(1.0, 5, nothing)

mutable struct ProjectionFullCore <: RPMProjectionType
    S::Union{Matrix{Float64}, Nothing}
end
ProjectionFullCore() = ProjectionFullCore(nothing)

function project!(type::ProjectionStdCore, x, q, b, iter::Int64)
    stdCore!(x, q, b, type.α)

    return nothing
end

function project!(type::ProjectionLowCore, x, q, b, iter::Int64)

    # Allocate space in the first iteration.
    if iter == 1
        d = length(x)
        type.Z = Vector{Float64}[zeros(Float64, d) for i in 1:type.m]
    end
    lowCore!(x, q, b, type.Z, type.α)

    return nothing
end

function project!(type::ProjectionFullCore, x, q, b, iter::Int64)

    if iter == 1
        d = length(x)
        type.S = diagm(ones(Float64, d))
    end
    fullCore!(x, q, b, type.S)

    return nothing
end

"""
    stdCore(x :: Vector{Float64}, q :: Vector{Float64}, b :: Float64; α :: Float64 = 1.0)

Implements a no memory Rank-one RPM method, which is given by

```math
x_{k+1} = x_k + \\alpha q ( b - q'x)
```

# Arguments

- `x::Vector{Float64}`, the current iterate
- `q::Vector{Float64}`, a row of the matrix
- `b::Float64`, the coefficient corresponding to the row `q`

# Keywords
- `α::Float64 = 1.0`, the step length

# Returns
"""
function stdCore!(
    x::Vector{Float64},
    q::Vector{Float64},
    b::Float64,
    α::Float64 = 1.0,
)
    x .= x + q*(α*(b - dot(q,x)))/dot(q,q)

    return nothing
end

"""
    lowCore(args...; kwargs...)

Arguments Implements low memory Rank-one RPM Method

# Arguments

- `x::Vector{Float64}`, current iterate
- `q::Vector{Float64}`, un-orthogonalized search direction
- `b::Float64`, coefficient corresponding to `q`
- `Z::Vector{Vector{Float64}}`, a collection of orthonormal vectors

# Keywords
- `α::Float64 = 1.0`, step length

# Returns
"""
function lowCore!(
    x::Vector{Float64},
    q::Vector{Float64},
    b::Float64,
    Z::Vector{Vector{Float64}},
    α::Float64 = 1.0,
)
    #Compute orthogonal component of q using modified Gram-Schmidt
    u = q
    for z in Z
        u = u - dot(z,u)*z
    end

    nrmU = norm(u)
    if nrmU < 1e-15 * length(x)
        return x, Z
    end

    #Update Iterate
    res = b - dot(q,x)
    x .= x + u*(α*res/dot(u,q))

    #Update Orthonormal set
    z = u/nrmU
    Z .= push!(Z[2:end],z)

    return nothing
end

"""
    fullCore(x :: Vector{Float64}, q :: Vector{Float64}, b :: Float64, S :: Matrix{Float64})

Implements full memory Rank-one RPM Method

# Arguments
- `x :: Vector{Float64}`, current iterate
- `q :: Vector{Float64}`, un-orthogonalized search direction
- `b :: Float64`, coefficient corresponding to `q`
- `S :: Matrix{Float64}`, an orthogonal projection matrix

# Returns
"""
function fullCore!(
    x :: Vector{Float64},
    q :: Vector{Float64},
    b :: Float64,
    S :: Matrix{Float64},
)
    #Compute orthogonal component of q using S (projection matrix)
    u = S*q

    if dot(u,u) < 1e-30 * length(x)
        return x, S
    end

    #Update Iterate
    res = (b - dot(q,x))
    γ = dot(u,q)
    x .= x + u*(res/γ)

    #Update Projection matrix
    S .= (I - (u/γ)*q')*S

    return nothing
end
