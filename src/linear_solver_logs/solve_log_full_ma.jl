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
mutable struct MAInfo
    lambda1::Int64
    lambda2::Int64
    lambda::Int64
    flag::Bool
    idx::Int64
    res_window::Vector{Float64}
end

mutable struct LSLogFullMA <: LinSysSolverLog
    collection_rate::Int64
    ma_info::MAInfo
    iota_hist::Vector{Float64}
    rho_hist::Vector{Float64}
    width_hist::Vector{Int64}
    resid_norm::Function
    iterations::Int64
    converged::Bool
end

LSLogFullMA() = LSLogFullMA(
                          1,
                          MAInfo(1, 30, 1, false, 1, zeros(30)),
                          Float64[], 
                          Float64[], 
                          Int64[],
                          norm, 
                          -1, 
                          false
                         )

LSLogFullMA(lambda2) = LSLogFullMA(
                          1,
                          MAInfo(1, lambda2, 1, false, 1, zeros(lambda2)),
                          Float64[], 
                          Float64[], 
                          Int64[],
                          norm, 
                          -1, 
                          false
                         )

LSLogFullMA(lambda1, lambda2) = LSLogFullMA(
                          1,
                          MAInfo(lambda1, lambda2, 1, false, 1, zeros(lambda1)),
                          Float64[], 
                          Float64[], 
                          Int64[],
                          norm, 
                          -1, 
                          false
                         )

#Function to update the moving average

function log_update!(
    log::LSLogFullMA,
    sampler::LinSysSampler,
    x::AbstractVector,
    samp::Tuple,
    iter::Int64,
    A::AbstractArray,
    b::AbstractVector
)
    ma_info = log.ma_info
    log.iterations = iter
    
    # Compute the current residual to second power to align with theory
    res::Float64 = size(samp[1],2) != 1 ? log.resid_norm(samp[1] * x - samp[2])^2 :  
                                          log.resid_norm(dot(samp[1], x) - samp[2])^2 
    
    # Check if MA is in lambda1 or lambda2 regime
    if ma_info.flag
        Update_MAEstimators!(log, ma_info, res, ma_info.lambda2, iter)
    else
        if res < ma_info.res_window[ma_info.idx]
            Update_MAEstimators!(log, ma_info, res, ma_info.lambda1, iter)
        else
            Update_MAEstimators!(log, ma_info, res, ma_info.lambda1, iter)
            ma_info.flag = true 
        end

    end

end

function Update_MAEstimators!(log, ma_info, res, lambda_base, iter)
    accum = 0
    accum2 = 0
    ma_info.idx = ma_info.idx < ma_info.lambda2 ? ma_info.idx + 1 : 1
    ma_info.res_window[ma_info.idx] = res
    #Check if entire storage buffer can be used
    if ma_info.lambda == lambda_base 
        # Compute the moving average
        @simd for i in 1:ma_info.lambda2
            accum += ma_info.res_window[i]
            accum2 += ma_info.res_window[i]^2
        end
       
        if mod(iter, log.collection_rate) == 0 || iter == 0
            push!(log.width_hist, ma_info.lambda)
            push!(log.rho_hist, accum / ma_info.lambda) 
            push!(log.iota_hist, accum2 / ma_info.lambda) 
        end
    else
        # Get the difference between the start and current lambda
        diff = ma_info.idx - ma_info.lambda
        
        # Determine start point for first loop
        startp1 = diff < 0 ? 1 : (diff + 1)
        
        # Determine start and endpoints for second loop
        startp2 = diff > 0 ? 2 : lambda_base + diff + 1  
        endp2 = diff > 0 ? 1 : lambda_base

        # Compute the moving average
        @simd for i in startp1:ma_info.idx
            accum += ma_info.res_window[i]
            accum2 += ma_info.res_window[i]^2
        end

        @simd for i in startp2:endp2
            accum += ma_info.res_window[i]
            accum2 += ma_info.res_window[i]^2
        end
        if mod(iter, log.collection_rate) == 0 || iter == 0
            push!(log.width_hist, ma_info.lambda)
            push!(log.rho_hist, accum / ma_info.lambda) 
            push!(log.iota_hist, accum2 / ma_info.lambda) 
        end
        ma_info.lambda += 1
    end

end
