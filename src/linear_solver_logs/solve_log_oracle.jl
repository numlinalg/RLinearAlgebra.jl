# This file is part of RLinearAlgebra.jl

# using LinearAlgebra

"""
    LSLogOracle <: LinSysSolverLog

A mutable structure that stores information about a randomized linear solver's behavior.
    The log assumes that the true solution of the system is known and will be supplied.
    The goal of this log is usually for research, development, or testing.

# Fields
- `solution::AbstractVector`, a solution to the given linear system.
- `collection_rate::Int64`, the frequency with which to record information to append to the
    remaining fields, starting with the initialization (i.e., iteration 0).
- `error_hist::Vector{Float64}`, retains a vector of numbers corresponding to the error
    between the iterates of the solver and the `solution`
- `error_norm::Function`, a function that accepts a single vector argument and returns a
    scalar. Used to compute the error size.
- `resid_hist::Vector{Float64}`, retains a vector of numbers corresponding to the residual
    (uses the whole system to compute the residual)
- `resid_norm::Function`, a function that accepts a single vector argument and returns a
    scalar. Used to compute the residual size.
- `iterations::Int64`, the number of iterations of the solver.
- `converged::Bool`, a flag to indicate whether the system has converged by some measure

# Constructors
- Calling `LSLogOracle(x_star::Vector{Float64})` sets `solution = x_star`,
    `collection_rate = 1`, `error_hist = Float64[]`, `error_norm = norm` (Euclidean norm),
    `resid_hist = Float64[]`, `resid_norm = norm` (Euclidean norm), `iterations=-1`,
    and `converged = false`.
- Calling `LSLogOracle(x_star::Vector{Float64}, cr::Int64)` sets the structure with the
    same parameters as for `LSLogOracle(x_star)` except `collection_rate = cr`.
"""
mutable struct LSLogOracle <: LinSysSolverLog
    solution::AbstractVector
    collection_rate::Int64
    error_hist::Vector{Float64}
    error_norm::Function
    resid_hist::Vector{Float64}
    resid_norm::Function
    iterations::Int64
    converged::Bool
end
LSLogOracle(x_star::Vector{Float64}) = LSLogOracle(x_star, 1, Float64[], norm, Float64[],
    norm, -1, false)
LSLogOracle(x_star::Vector{Float64}, cr::Int64) = LSLogOracle(x_star, cr, Float64[], norm,
    Float64[],norm, -1, false)

# Common interface for update
function log_update!(
    log::LSLogOracle,
    sampler::LinSysSampler,
    x::AbstractVector,
    samp::Tuple,
    iter::Int64,
    A::AbstractArray,
    b::AbstractVector
)
    # Update iteration counter
    log.iterations = iter

    # Check whether to record information
    if mod(iter, 1:log.collection_rate) == log.collection_rate
        # Push absolute error between iterate and solution
        push!(log.error_hist, log.error_norm(x - log.solution))

        # Push residual norm of iterate
        push!(log.resid_hist, log.resid_norm(A * x - b))
    end

    return nothing
end
