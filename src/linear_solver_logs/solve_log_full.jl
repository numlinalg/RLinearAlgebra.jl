# This file is part of RLinearAlgebra.jl

# Implement moving average by using the structs and functions in another file. 
include("solve_log_ma.jl")

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
    ma_info::MAInfo  # Implement moving average (MA)
    lambda_hist::Vector{Int64}  # Implement moving average (MA)
    iota_hist::Vector{Float64}  # Implement moving average (MA)
end
LSLogFull() = LSLogFull(1, Float64[], norm, -1, false, 
                        MAInfo(lambda1, lambda2, lambda1, false, 1, zeros(lambda2)),
                        Int64[], 
                        Float64[])
LSLogFull(cr::Int64) = LSLogFull(cr, Float64[], norm, -1, false,
                        MAInfo(lambda1, lambda2, lambda1, false, 1, zeros(lambda2)),
                        Int64[], 
                        Float64[])

# Common interface for update
function log_update!(
    log::LSLogFull,
    sampler::LinSysSampler,
    x::AbstractVector,
    samp::Tuple,
    iter::Int64,
    A::AbstractArray,
    b::AbstractVector
)
    # Update iteration counter
    log.iterations = iter

    ###############################
    # Implement moving average (MA)
    ###############################
    ma_info = log.ma_info 
    # Compute the current residual to second power to align with theory
    res_norm_iter =  log.resid_norm(A * x - b)
    res::Float64 = res_norm_iter^2

    # Check if MA is in lambda1 or lambda2 regime
    if ma_info.flag
        update_ma!(log, res, ma_info.lambda2, iter)
    else
        # Check if we can switch between lambda1 and lambda2 regime
        # If it is in the monotonic decreasing of the sketched residual then we are in a lambda1 regime
        # otherwise we switch to the lambda2 regime which is indicated by the changing of the flag
        # because update_ma changes res_window and ma_info.idx we must check condition first
        flag_cond = iter == 0 || res <= ma_info.res_window[ma_info.idx] 
        update_ma!(log, res, ma_info.lambda1, iter)
        ma_info.flag = !flag_cond

    end

    ###############################

    # Check whether to record information
    if mod(iter, 1:log.collection_rate) == log.collection_rate

        # Push residual norm of current iterate
        push!(log.resid_hist, res_norm_iter)
    end

    return nothing
end
