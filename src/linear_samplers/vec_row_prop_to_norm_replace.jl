# This file is pat of RLinearAlgebra.jl

"""
    LinSysVecRowPropToNormSampler <: LinSysVecRowSampler

A mutable structure that specifies sampling from the rows of the equation where
the probability of selecting a given equation is proportional to the sum of squares of the
coefficients of the given equation. The solver will appropriately initialize the
distribution.

See Strohmer, T., Vershynin, R. A Randomized Kaczmarz Algorithm with Exponential
Convergence. J Fourier Anal Appl 15, 262 (2009). https://doi.org/10.1007/s00041-008-9030-4

# Aliases
- `LinSysVecRowSVSampler`

# Fields
- `dist::Vector{Float64}`, probability vector representing a distribution over rows.

Calling `LinSysVecRowPropToNormSampler()` or `LinSysVecRowSVSampler()` defaults `dist` to
`[1.0]`.

!!! note
    When `iter == 1`, the vector `dist` will be overwritten to ensure the correct probability
    weights. 
"""
mutable struct LinSysVecRowPropToNormSampler <: LinSysVecRowSampler
    dist::Vector{Float64}
end
LinSysVecRowSVSampler = LinSysVecRowPropToNormSampler
LinSysVecRowPropToNormSampler() = LinSysVecRowPropToNormSampler([1.0])

# Common sample interface for linear systems
function sample(
    type::LinSysVecRowSVSampler,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    if iter == 1
        type.dist = RLinearAlgebra.frobenius_norm_distribution(A, true)
    end

    eqn_ind = StatsBase.sample(1:length(type.dist), Weights(type.dist))

    return A[eqn_ind, :], b[eqn_ind]
end
