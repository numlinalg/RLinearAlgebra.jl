# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type
# 4. Tests

# using Random, LinearAlgebra

"""
    LinSysVecColDetermCyclic <: LinSysVecColSelect

An immutable structure without any fields. Specifies deterministic cycling through the
columns of a linear system.
"""
struct LinSysVecColDetermCyclic <: LinSysVecColSelect end

# Common sample interface for linear systems
function sample(
    type::LinSysVecColDetermCyclic,
    A::Matrix{Float64},
    b::Vector{Float64},
    x::Vector{Float64},
    iter::Int64
)

    col_ind = mod(iter, 1:length(x))

    # Search direction
    v = zeros(length(x))
    v[col_ind] = 1.0

    # Normal equation residual
    res = dot(A[:,col_ind], A * x - b)

    return v, A, res
end
#export LinSysVecColDetermCyclic
