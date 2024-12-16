# This file is part of RLinearAlgebra.jl

# Implement moving average by using the structs in the file that implementing MA. 
include("solve_log_ma.jl")

"""
    LSLogFullMA <: LinSysSolverLog

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
- Calling `LSLogFullMA()` sets `collection_rate = 1`, `resid_hist = Float64[]`,
    `resid_norm = norm` (Euclidean norm), `iterations = -1`, and `converged = false`.
- Calling `LSLogFullMA(cr::Int64)` is the same as calling `LSLogFullMA()` except
    `collection_rate = cr`.
"""
mutable struct LSLogFullMA <: LinSysSolverLog
    collection_rate::Int64
    resid_hist::Vector{Float64}
    resid_norm::Function
    iterations::Int64
    converged::Bool
    ma_info::MAInfo  # Implement moving average (MA)
    lambda_hist::Vector{Int64}  # Implement moving average (MA)
    iota_hist::Vector{Float64}  # Implement moving average (MA)
end

LSLogFullMA(;
            collection_rate = 1, 
            lambda1 = 1, 
            lambda2 = 30, 
           ) = LSLogFullMA( collection_rate, 
                            Float64[], 
                            norm, 
                            -1, 
                            false, 
                            MAInfo(lambda1, lambda2, lambda1, false, 1, zeros(lambda2)),
                            Int64[], 
                            Float64[]
                          )

# Common interface for update
function log_update!(
    log::LSLogFullMA,
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

end

"""
    update_ma!(
        log::LSLogMA, 
        res::Union{AbstractVector, Real}, 
        lambda_base::Int64, 
        iter::Int64
    ) 

Function that updates the moving average tracking statistic. 

# Inputs
- `log::LSLogMA`, the moving average log structure.
- `res::Union{AbstractVector, Real}`, the sketched residual for the current iteration. 
- `lambda_base::Int64`, which lambda, between lambda1 and lambda2, is currently being used.
- `iter::Int64`, the current iteration.

# Outputs
Updates the log datatype and does not explicitly return anything.
"""
function update_ma!(log::LSLogFullMA, res::Union{AbstractVector, Real}, lambda_base::Int64, iter::Int64)
    # Variable to store the sum of the terms for rho
    accum = 0
    # Variable to store the sum of the terms for iota 
    accum2 = 0
    ma_info = log.ma_info
    ma_info.idx = ma_info.idx < ma_info.lambda2 && iter != 0 ? ma_info.idx + 1 : 1
    ma_info.res_window[ma_info.idx] = res
    #Check if entire storage buffer can be used
    if ma_info.lambda == ma_info.lambda2 
        # Compute the moving average
        for i in 1:ma_info.lambda2
            accum += ma_info.res_window[i]
            accum2 += ma_info.res_window[i]^2
        end
        
        if mod(iter, log.collection_rate) == 0 || iter == 0
            push!(log.lambda_hist, ma_info.lambda)
            push!(log.resid_hist, accum / ma_info.lambda) 
            push!(log.iota_hist, accum2 / ma_info.lambda) 
        end

    else
        # Consider the case when lambda <= lambda1 or  lambda1 < lambda < lambda2
        diff = ma_info.idx - ma_info.lambda
        # Because the storage of the residual is based dependent on lambda2 and 
        # we want to sum only the previous lamdda terms we could have a situation
        # where we want the first `idx` terms of the buffer and the last `diff`
        # terms of the buffer. Doing this requires two loops
        # If `diff` is negative there idx is not far enough into the buffer and
        # two sums will be needed
        startp1 = diff < 0 ? 1 : (diff + 1)
        
        # Assuming that the width of the buffer is lambda2 
        startp2 = diff < 0 ? ma_info.lambda2 + diff + 1 : 2 
        endp2 = diff < 0 ? ma_info.lambda2 : 1

        # Compute the moving average two loop setup required when lambda < lambda2
        for i in startp1:ma_info.idx
            accum += ma_info.res_window[i]
            accum2 += ma_info.res_window[i]^2
        end

        for i in startp2:endp2
            accum += ma_info.res_window[i]
            accum2 += ma_info.res_window[i]^2
        end

        #Update the log variable with the information for this update
        if mod(iter, log.collection_rate) == 0 || iter == 0
            push!(log.lambda_hist, ma_info.lambda)
            push!(log.resid_hist, accum / ma_info.lambda) 
            push!(log.iota_hist, accum2 / ma_info.lambda) 
        end
        
        ma_info.lambda += ma_info.lambda < lambda_base ? 1 : 0
    end

end



