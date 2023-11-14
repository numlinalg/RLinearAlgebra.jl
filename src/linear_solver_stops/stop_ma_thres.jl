# This file is part of RLinearAlgebra.jl

"""
    LSStopMA <: LinSysStopCriterion

A structure that specifies a stopping criterion that incoroporates the randomness of the moving average estimator. That is, once a method
    achieves a certain number of iterations, it stops.

# Fields
- `max_iter::Int64`, the maximum number of iterations.
- `threshold::Float64`, the value of the estimator that is sufficient progress. 
- `delta1::Float64`, the percent below the threshold does the true value of the progress estimator need to be for not stopping to be a 
    mistake. This is equivalent to stopping too late.
- `delta2::Float64`, the percent above the threshold does the true value of the progress estimator need to be for stopping to be a 
    mistake. This is equivalent to stopping too early.
- `chi1::Float64`, the probability that the stopping too late action occurs.
- `chi2::Float64`, the probability that the stopping too early action occurs.
"""
struct LSStopMA <: LinSysStopCriterion
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
    log::LSLogFullMA,
    stop::LSStopMA
)
    its = log.iterations
    iota_threshold = variance_threshold(log)
    return (sqrt(log.iota_hist[its]) <= iota_threshold && 
            log.iota_hist[its] <= threshold) ||  its == stop.max_iter ? true : false
end

# Once the sigma2 is known function computest the threshold
function iota_threshold(log::LSLogFullMA, stop::LSStopMA)
    delta1 = stop.delta1
    delta2 = stop.delta2
    chi1 = stop.chi1
    chi2 = stop.chi2
    sigma2 = log.sigma2
    # If the constants for the sub-Exponential distribution are not defined then define them
    if log.sigma2 <: Nothing
        get_SE_constants!(log, log.sampler)
    end
    
    #If there is an omega in the sub-Exponential distribution then skip that calculation 
    if log.omega <: Nothing
        c = min((1 - delta1)^2 * threshold^2 / (2 * log(1/chi1)), (delta2 - 1)^2 * threshold^2 / (2 * log(1/chi2)))
        c /= (sigma2 * sqrt(log.iota_hist[log.iterations]))
    else
        siota = sqrt(log.iota_hist[log.iterations])
        min1 = min((1 - delta1)^2 * threshold^2 / (2 * log(1/chi1) * c * siota), lambda * (1 - delta1) * threshold / (2 * log(1/chi1) * log.omega))
        min2 = min((delta2 - 1)^2 * threshold^2 / (2 * log(1/chi2) * c * siota), lambda * (delta2 - 1) * threshold / (2 * log(1/chi2) * log.omega))
        c = min(min1, min2) 
    end

    return c 

end

# Get the sub-Exponential constants for each of the samplers
function get_SE_constants!(log::LSLogFullMA, sampler::LinSysVecRowDetermCyclic)
    lambda = log.MAInfo.lambda
    log.sigma2 = log.max_dimension^2 * (1 + log(lambda)) / (4 * log.eta * lambda)
end


function get_SE_constants!(log::LSLogFullMA, sampler::LinSysVecColOneRandCyclic)  
    lambda = log.MAInfo.lambda
    log.sigma2 = log.max_dimension^2 * (1 + log(lambda)) / (4 * log.eta * lambda)
end
