# This file is part of RLinearAlgebra.jl

# using LinearAlgebra

"""
    LSLogFull <: LinSysSolverLog

A mutable structure that stores information about a randomized linear solver's behavior.
    The log assumes that the full linear system is available for processing. The goal of
    this log is usually for research, development or testing as it is unlikely that the
    entire residual vector is readily available.

# Fields
- `collection_rate::Int64`, the frequency with which to record information to append to the
    remaining fields, starting with the initialization (i.e., iteration 0).
- `resid_hist::Vector{Float64}`, retains a vector of numbers corresponding to the residual
    (uses the whole system to compute the residual)
- `resid_norm::Function`, a function that accepts a single vector argument and returns a
    scalar. Used to compute the residual size.
- `iterations::Int64`, the number of iterations of the solver.
- `converged::Bool`, a flag to indicate whether the system has converged by some measure

# Constructors
- Calling `LSLogFull()` sets `collection_rate = 1`, `resid_hist = Float64[]`,
    `resid_norm = norm` (Euclidean norm), `iterations = -1`, and `converged = false`.
- Calling `LSLogFull(cr::Int64)` is the same as calling `LSLogFull()` except
    `collection_rate = cr`.
"""
mutable struct LSLogFull <: LinSysSolverLog
    collection_rate::Int64
    resid_hist::Vector{Float64}
    resid_norm::Function
    iterations::Int64
    converged::Bool
end
LSLogFull() = LSLogFull(1, Float64[], norm, -1, false)
LSLogFull(cr::Int64) = LSLogFull(cr, Float64[], norm, -1, false)

# Common interface for update
function log_update!(
    log::LSLogFull,
    sampler::S where S<:LinSysSampler,
    x::Vector{Float64},
    samp::T where T<:Tuple,
    iter::Int64,
    A,
    b
)
    # Update iteration counter
    log.iterations = iter

    # Check whether to record information
    if mod(iter, 1:log.collection_rate) == log.collection_rate

        # Push residual norm of current iterate
        push!(log.resid_hist, log.resid_norm(A * x - b))
    end

    return nothing
end
