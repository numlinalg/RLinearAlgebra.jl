# This file is pat of RLinearAlgebra.jl
# 1. Specifies type
# 2. Implements sample function
# 3. Exports Type
"""
    LinSysVecRowGaussSampler <: LinSysVecRowSampler

An immutable structure without fields that specifies taking a linear combination of all
equations with the coefficients being independent Gaussian random variables.
"""
struct LinSysVecRowGaussSampler <: LinSysVecRowSampler end

# Common sample interface for linear systems
function sample(
    type::LinSysVecRowGaussSampler,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    coef = randn(length(b))

    return A'*coef, dot(b,coef)
end
