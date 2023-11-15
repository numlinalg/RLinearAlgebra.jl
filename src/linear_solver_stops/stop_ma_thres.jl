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
    log::LSLogFullMA,
    stop::LSStopMA
)
    its = log.iterations
    if its > 0
        I_threshold = iota_threshold(log, stop)
        thresholdChecks = sqrt(log.iota_hist[its]) <= I_threshold && 
        log.resid_hist[its] <= stop.threshold
    else
        thresholdChecks = false
    end
    return (thresholdChecks || its == stop.max_iter ? true : false)
end

# Once the sigma2 is known function computest the threshold
function iota_threshold(hist::LSLogFullMA, stop::LSStopMA)
    delta1 = stop.delta1
    delta2 = stop.delta2
    chi1 = stop.chi1
    chi2 = stop.chi2
    threshold = stop.threshold
    lambda = hist.ma_info.lambda
    # If the constants for the sub-Exponential distribution are not defined then define them
    if typeof(hist.sigma2) <: Nothing
        get_SE_constants!(hist, hist.sampler)
    end
    
    #If there is an omega in the sub-Exponential distribution then skip that calculation 
    if typeof(hist.omega) <: Nothing
        # Compute the threshold bound in the case where there is no omega
        c = min((1 - delta1)^2 * threshold^2 / (2 * log(1/chi1)), (delta2 - 1)^2 * threshold^2 / (2 * log(1/chi2)))
        c /= (hist.sigma2 * sqrt(hist.iota_hist[hist.iterations])) * (1 + log(lambda)) / lambda
    else
        #compute error bound when there is an omega
        siota = (hist.sigma2 * sqrt(hist.iota_hist[hist.iterations])) * (1 + log(lambda)) / lambda
        min1 = min((1 - delta1)^2 * threshold^2 / (2 * log(1/chi1) * siota),
                   lambda * (1 - delta1) * threshold / (2 * log(1/chi1) * hist.omega))
        min2 = min((delta2 - 1)^2 * threshold^2 / (2 * log(1/chi2) * siota),
                   lambda * (delta2 - 1) * threshold / (2 * log(1/chi2) * hist.omega))
        c = min(min1, min2) 
    end

    return c 

end

# Get the sub-Exponential constants for each of the samplers.
# For the direct row samplers the constants will be just nrows^2 / (4 eta),
# where eta is a user controlled parameter to tighten the variance bound.

for type in (LinSysVecRowDetermCyclic,LinSysVecRowHopRandCyclic,LinSysVecRowPropToNormSampler,
             LinSysVecRowSVSampler, LinSysVecRowUnidSampler,
             LinSysVecRowUnifSampler, LinSysVecRowSparseUnifSampler,
             LinSysVecRowOneRandCyclic, LinSysVecRowDistCyclic,
             LinSysVecRowResidCyclic, LinSysVecRowMaxResidual,
             LinSysVecRowMaxDistance)
    @eval begin
        function get_SE_constants!(log::LSLogFullMA, sampler::Type{$type})
            log.sigma2 = log.max_dimension^2 / (4 * log.eta)
        end

    end

end


#Column subsetting methods have same constants as in row case
for type in (LinSysVecColOneRandCyclic, LinSysVecColDetermCyclic)
    @eval begin
        function get_SE_constants!(log::LSLogFullMA, sampler::Type{$type})
            log.sigma2 = log.max_dimension^2 / (4 * log.eta)
        end

    end

end

# For row samplers with gaussian sampling we have sigma2 = 1/.2345 and omega = .1127
for type in (LinSysVecRowGaussSampler, LinSysVecRowSparseGaussSampler)
    @eval begin
        function get_SE_constants!(log::LSLogFullMA, sampler::Type{$type})
            log.sigma2 = 1 / (0.2345 * log.eta)
            log.omega = .1127
        end

    end

end
