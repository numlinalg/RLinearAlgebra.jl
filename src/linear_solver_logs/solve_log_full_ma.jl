# This file is part of RLinearAlgebra.jl

# using LinearAlgebra

mutable struct MAInfo
    lambda1::Int64
    lambda2::Int64
    lambda::Int64
    flag::Bool
    idx::Int64
    res_window::Vector{Float64}
end

"""
    LSLogFullMA <: LinSysSolverLog

A mutable structure that stores information about a randomized linear solver's behavior.
    The log assumes that only a random block of full linear system is available for processing. 
    The goal of this log is usually for all use cases as it acts as a cheap approximation of the 
    full residual.

# Fields
- `collection_rate::Int64`, the frequency with which to record information to append to the
    remaining fields, starting with the initialization (i.e., iteration 0).
- `ma_info::MAInfo`, structure that holds information relevant only to moving average, 
   like the two choices of moving average widths (lambda1 and lambda2), the current moving average
   width (lambda), a flag to indicate which moving average regime, between lambda1 and lambda2, is 
   being used, a vector containing the values to be averaged (res_window), and an indicator of where 
   the next value should be placed (idx).
- `resid_hist::Vector{Float64}`, a structure that contains the moving average of the error proxy, 
   typically the norm of residual or gradient of normal equations.
- `iota_hist::Vector{Float64}`, a structure that contains the moving average of the error proxy, 
   typically the norm of residual or gradient of normal equations.
- `width_hist::Vector{Int64}`, data structure that contains the widths used in the moving average 
   calculation at each iteration.
- `resid_norm::Function`, a function that accepts a single vector argument and returns a
    scalar. Used to compute the residual size.
- `iterations::Int64`, the number of iterations of the solver.
- `converged::Bool`, a flag to indicate whether the system has converged by some measure.
- `sampler::DataType`, a data type that is needed for computing constants used in the uncertainty
  quantification and stopping steps. This is updated with each `log_update!` call. 
- `max_dimension::Int64`, a value that stores the max between the row and column dimension needed for
  computation of stopping criterion and uncertainty sets.
- `sigma2::Union{Float64, Nothing}`, a value that stores the sigma^2 parameter of a sub-Exponential
  distribution used for determining stopping criterion and uncertainty sets.
- `omega::Union{Float64, Nothing}`, a value that stores the omega parameter of a sub-Exponential
  distribution used for determining stopping criterion and uncertainty sets.
- `eta::Float64, a parameter that allows the adjustment of the uncertainty quantification if the 
  size of the default covariance is too wide for the particular problem.
- `true_res`, a boolean indicating if we want the true residual computed instead of approximate.
# Constructors
- Calling `LSLogFullMA()` sets `collection_rate = 1`,  `lambda1 = 1`,
    `lambda2 = 30`, `resid_hist = Float64[]`, `iota_hist = Float64[]`, `width_hist = Int64[]`, 
    `resid_norm = norm` (Euclidean norm), `iterations = -1`, `converged = false`, 
    `sampler = LinSysVecRowDetermCyclic`, `max_dimension = 0`, `sigma2 = nothing`, `omega = nothing`,
    `eta = 1`, and `true_res = false`. The user can specify their own values of lambda1, lambda2, sigma2, omega, and eta using 
    key word arguments as inputs into the constructor.
"""
mutable struct LSLogFullMA <: LinSysSolverLog
    collection_rate::Int64
    ma_info::MAInfo
    resid_hist::Vector{Float64}
    iota_hist::Vector{Float64}
    width_hist::Vector{Int64}
    resid_norm::Function
    iterations::Int64
    converged::Bool
    sampler::DataType
    max_dimension::Int64
    sigma2::Union{Float64, Nothing}
    omega::Union{Float64, Nothing}
    eta::Float64
    true_res::Bool
end

LSLogFullMA() = LSLogFullMA(
                          1,
                          MAInfo(1, 30, 1, false, 1, zeros(30)),
                          Float64[], 
                          Float64[], 
                          Int64[],
                          norm, 
                          -1, 
                          false,
                          LinSysVecRowDetermCyclic,
                          0,
                          nothing,
                          nothing,
                          1,
                          false
                         )

LSLogFullMA(;lambda1 = 1, lambda2 = 30, sigma2 = nothing, omega = nothing, eta = 1, true_res = false) = LSLogFullMA(
                          1,
                          MAInfo(1, lambda2, 1, false, 1, zeros(lambda2)),
                          Float64[], 
                          Float64[], 
                          Int64[],
                          norm, 
                          -1, 
                          false,
                          LinSysVecColDetermCyclic,
                          0,
                          sigma2,
                          omega,
                          eta, 
                          true_res
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
    log.max_dimension = maximum(size(A))
    ma_info = log.ma_info
    log.iterations = iter
    log.sampler = typeof(sampler)  
    if iter != 0
        #Check if we want exact residuals computed
        if !log.true_res
            # Compute the current residual to second power to align with theory
            res::Float64 = size(samp[1],2) != 1 ? log.resid_norm(samp[1] * x - samp[2])^2 :  
                                          log.resid_norm(dot(samp[1], x) - samp[2])^2 
        else 
            res = log.resid_norm(A * x - b)^2 
        end
        # Check if MA is in lambda1 or lambda2 regime
        if ma_info.flag
            Update_MAEstimators!(log, ma_info, res, ma_info.lambda2, iter)
        else
            #Check if we can switch between lambda1 and lambda2 regime
            if res < ma_info.res_window[ma_info.idx]
                Update_MAEstimators!(log, ma_info, res, ma_info.lambda1, iter)
            else
                Update_MAEstimators!(log, ma_info, res, ma_info.lambda1, iter)
                ma_info.flag = true 
            end

        end

    end

end

# Update the moving average estimator requires the log variable, ma_info,
# observed residual, and a lambda_base which corresponds to which lambda regime we are in. 
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
            push!(log.resid_hist, accum / ma_info.lambda) 
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

        # Compute the moving average two loop setup required when lambda < lambda2
        @simd for i in startp1:ma_info.idx
            accum += ma_info.res_window[i]
            accum2 += ma_info.res_window[i]^2
        end

        @simd for i in startp2:endp2
            accum += ma_info.res_window[i]
            accum2 += ma_info.res_window[i]^2
        end

        #Update the log variable with the information for this update
        if mod(iter, log.collection_rate) == 0 || iter == 0
            push!(log.width_hist, ma_info.lambda)
            push!(log.resid_hist, accum / ma_info.lambda) 
            push!(log.iota_hist, accum2 / ma_info.lambda) 
        end
        
        ma_info.lambda += 1
    end

end

#Function that will return rho and its uncertainty from a LSLogFullMA type 
"""
    get_uncertainty(log::LSLogFullMA; alpha = .95)
    A function that takes a LSLogFullMA type and a confidence level, alpha, and returns credible intervals for for every rho in the log, specifically it returns a tuple with (rho, Upper bound, Lower bound).
"""
function get_uncertainty(hist::LSLogFullMA; alpha = .95)
    lambda = hist.ma_info.lambda
    l = length(hist.iota_hist)
    upper = zeros(l)
    lower = zeros(l)
    # If the constants for the sub-Exponential distribution are not defined then define them
    if typeof(hist.sigma2) <: Nothing
        get_SE_constants!(hist, hist.sampler)
    end
    
    for i in 1:l
        width = hist.width_hist[i]
        iota = hist.iota_hist[i]
        rho = hist.resid_hist[i]
        #Define the variance term for the Gaussian part
        cG = hist.sigma2 * (1 + log(width)) * iota / (width * hist.eta)
        #If there is an omega in the sub-Exponential distribution then skip that calculation 
        if typeof(hist.omega) <: Nothing
            # Compute the threshold bound in the case where there is no omega
            diffG = sqrt(cG * 2 * log(2/(1-alpha)))
            upper[i] = rho + diffG
            lower[i] = rho - diffG
        else
            #compute error bound when there is an omega
            diffG = sqrt(cG * 2 * log(2/(1-alpha)))
            diffO = sqrt(iota) * 2 * log(2/(1-alpha)) * hist.omega / (hist.eta * width)
            diffM = min(diffG, diffO)
            upper[i] = rho + diffG
            lower[i] = rho - diffG
        end
        
    end

    return (hist.resid_hist, upper, lower)  

end

