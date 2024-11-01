# This file is part of RLinearAlgebra.jl

"""
    LSStopMA <: LinSysStopCriterion

A structure that specifies a stopping criterion that incoroporates the randomness of the moving average estimator. That is, once a method
    achieves a certain number of iterations, it stops.

# Fields
- `max_iter::Int64`, the maximum number of iterations.
- `threshold::Float64`, the value of the estimator that is sufficient progress. 
- `delta1::Float64`, the percent below the threshold does the true value of the progress estimator need to be for not stopping to be a mistake. This is equivalent to stopping too late.
- `delta2::Float64`, the percent above the threshold does the true value of the progress estimator need to be for stopping to be a 
    mistake. This is equivalent to stopping too early.
- `chi1::Float64`, the probability that the stopping too late action occurs.
- `chi2::Float64`, the probability that the stopping too early action occurs.
# Constructors
- Calling LSStopMA(iter) will specify the users desired maximum number of iterations, threshold = 1e-10, delta1 = .9, delta2 = 1.1, chi1 = 0.01, and chi2 = 0.01.
"""
mutable struct LSStopMA <: LinSysStopCriterion
    max_iter::Int64
    threshold::Float64
    delta1::Float64
    delta2::Float64
    chi1::Float64
    chi2::Float64
end

LSStopMA(iter; 
         threshold = 1e-10,
         delta1 = .9,
         delta2 = 1.1,
         chi1 = .01,
         chi2 = .01
        ) = LSStopMA(iter, threshold, delta1, delta2, chi1, chi2)

# Common interface for stopping criteria
function check_stop_criterion(
    log::LSLogMA,
    stop::LSStopMA
)
    its = log.iteration
    if its > 0
        I_threshold = iota_threshold(log, stop)
        thresholdChecks = sqrt(log.iota_hist[its]) <= I_threshold && 
        log.resid_hist[its] <= stop.threshold
    else
        thresholdChecks = false
    end
    return (thresholdChecks || its == stop.max_iter ? true : false)
end

"""
    iota_threshold(hist::LSLogMA, stop::LSStopMA)

Function that computes the stopping criterion using the sub-Exponential distribution from the `LSLogMA`, and the stopping criterion information in `LSSopMA`. This function is not exported and thus not directly callable by the user.

# Inputs
- `hist::LSLogMA`, the log information for the moving average tracking.
- `stop::LSStopMA`, the stopping information for the stopping criterion.

# Ouputs
Returns the stoppping criterion value.

Pritchard, Nathaniel, and Vivak Patel. "Solving, Tracking and Stopping Streaming Linear Inverse Problems." arXiv preprint arXiv:2201.05741 (2024).
"""
function iota_threshold(hist::LSLogMA, stop::LSStopMA)
    delta1 = stop.delta1
    delta2 = stop.delta2
    chi1 = stop.chi1
    chi2 = stop.chi2
    threshold = stop.threshold
    lambda = hist.ma_info.lambda
    # If the constants for the sub-Exponential distribution are not defined then define them
    
    if typeof(hist.dist_info.sigma2) <: Nothing
        get_SE_constants!(hist, hist.dist_info.sampler)
    end
    #If there is an omega in the sub-Exponential distribution then skip that calculation 
    if typeof(hist.dist_info.omega) <: Nothing
        # Compute the threshold bound in the case where there is no omega
        c = min((1 - delta1)^2 * threshold^2 / (2 * log(1/chi1)), (delta2 - 1)^2 * 
                threshold^2 / (2 * log(1/chi2)))
        c /= (hist.dist_info.sigma2 * sqrt(hist.iota_hist[hist.iteration])) * (1 + log(lambda)) / lambda
    else
        #compute error bound when there is an omega
        siota = (hist.dist_info.sigma2 * sqrt(hist.iota_hist[hist.iteration])) * (1 + log(lambda)) / lambda
        min1 = min((1 - delta1)^2 * threshold^2 / (2 * log(1/chi1) * siota),
                   lambda * (1 - delta1) * threshold / (2 * log(1/chi1) * hist.dist_info.omega))
        min2 = min((delta2 - 1)^2 * threshold^2 / (2 * log(1/chi2) * siota),
                   lambda * (delta2 - 1) * threshold / (2 * log(1/chi2) * hist.dist_info.omega))
        c = min(min1, min2) 
    end

    return c 

end

