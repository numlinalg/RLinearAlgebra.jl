"""
    RPMProjectionType

Abstract row projection matrix encapsulation.
"""
abstract type RPMProjectionType end

"""
    ProjectionStdCore

A mutable subtype of `RPMProjectionType` that represents a standard row projection method.
The type can be constructed with an optional `Float64` argument for the relaxation parameter
(usually between `0.0` and `2.0`).

Calling `ProjectionStdCore()` defaults the relaxation parameter to `1.0`.
"""
mutable struct ProjectionStdCore <: RPMProjectionType
    α::Float64
end
ProjectionStdCore() = ProjectionStdCore(1.0)

"""
    ProjectionLowCore

A mutable subtype of `RPMProjectionType` that represents the standard row projection method
with orthogonalization of the current projection against a set of `m` previous projection
directions, where `m` is specified by the user.

# Fields

- `α::Float64`, the relaxation parameter (usually between `0.0` and `2.0`)
- `m::Int64`, the number of previous directions against which to orthogonalize
- `Z::Union{Vector{Vector{Float64}}, Nothing}`, stores the `m` vectors against which the
    current projection is orthogonalized against

Calling `ProjectionLowCore()` defaults to `ProjectionLowCore(1.0, 5, nothing)`.
"""
mutable struct ProjectionLowCore <: RPMProjectionType
    α::Float64
    m::Int64
    Z::Union{Vector{Vector{Float64}}, Nothing}
end
ProjectionLowCore() = ProjectionLowCore(1.0, 5, nothing)


"""
    `ProjectionFullCore`

A mutable subtype of `RPMProjectionType` that represents the standard row projection method
with full orthogonalization against all previous projections. Equivalently, this type
represents a solver for incrementally constructed matrix sketches.

# Fields

- `S::Union{Matrix{Float64}, Nothing}` is a matrix used for orthogonalizing against
    all previous search directions.

Calling `ProjectionFullCore()` defaults to `ProjectionFullCore(nothing)`.
"""
mutable struct ProjectionFullCore <: RPMProjectionType
    S::Union{Matrix{Float64}, Nothing}
end
ProjectionFullCore() = ProjectionFullCore(nothing)

"""
    project!(type::T, x, q, b, iter::Int64) where T<:RPMProjectionType

A wrapper function for updating the itertate `x` using an `RPMProjectionType` for a
hyperplane specified by a vector `q` and scalar `b`,

```math
\lbrace z : \langle q, z \rangle = b \rbrace.
```

The `iter` argument is the iteration counter.
"""
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
    stdCore!(x::Vector{Float64}, q::Vector{Float64}, b::Float64; α::Float64=1.0)

Implements an in-place vector row projection to update `x` according to the formula

```math
x_{k+1} = x_k + \\alpha q ( b - \langle q, x_k \rangle)/ \Vert q \Vert_2^2
```

# Arguments

- `x::Vector{Float64}`, the current iterate
- `q::Vector{Float64}`, a vector that defines a hyperplane
- `b::Float64`, a scalar used to define a hyperplane `q`

# Keywords
- `α::Float64`, the relaxation parameter between `0.0` and `2.0`. Defaults to `1.0`

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
    lowCore!(x::Vector{Float64}, q::Vector{Float64}, b::Float64, Z::Vector{Vector{Float64}};
        α::Float64=1.0)

Implements an in-place vector row projection to update `x` according to the formula

```math
x_{k+1} = x_k + \\alpha u ( b- \langle q, x_k \rangle ) / \langle q, u \rangle,
```
where `u` is the vector produced by orthogonalizing `q` against the set of vectors in `Z`.
Also updates the set `Z` with the vector `u` and expels the initial element of `Z`

# Arguments

- `x::Vector{Float64}`, current iterate
- `q::Vector{Float64}`, un-orthogonalized search direction
- `b::Float64`, coefficient corresponding to `q`
- `Z::Vector{Vector{Float64}}`, a collection of orthonormal vectors

# Keywords
- `α::Float64`, a relaxation parameter. Defaults to $1.0$.
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
    fullCore!(x::Vector{Float64}, q::Vector{Float64}, b::Float64, S::Matrix{Float64})

Implements an in-place row projection update to `x` according to the formulate

```math
x_{k+1} = x_k + \\alpha u ( b- \langle q, x_k \rangle ) / \langle q, u \rangle,
```
where `u` is the product of `S` and `q`. The matrix `S` is also updated according to

```math
S_{k+1} = \left( I - \frac{1}{\langle q, u \rangle} u q' ) S_k.
```

# Arguments

- `x :: Vector{Float64}`, current iterate
- `q :: Vector{Float64}`, un-orthogonalized search direction
- `b :: Float64`, coefficient corresponding to `q`
- `S :: Matrix{Float64}`, an orthogonal projection matrix

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
