"""
    MAStop

A structure that specifies a stopping criterion that incoroporates the randomness of the moving average estimator. That is, once a method
    achieves a certain number of iterations, it stops.

# Fields
- `threshold::Union{AbstractFloat, Tuple}`, the value of the estimator that is sufficient progress. 
- `delta1::AbstractFloat`, the percent below the threshold does the true value of the progress estimator need to be for not stopping to be a mistake. This is equivalent to stopping too late.
- `delta2::AbstractFloat`, the percent above the threshold does the true value of the progress estimator need to be for stopping to be a 
    mistake. This is equivalent to stopping too early.
- `chi1::AbstractFloat`, the probability that the stopping too late action occurs.
- `chi2::AbstractFloat`, the probability that the stopping too early action occurs.
# Constructors
- Calling MAStop(iter) will specify the users desired maximum number of iterations, threshold = 1e-10, delta1 = .9, delta2 = 1.1, chi1 = 0.01, and chi2 = 0.01.
"""
struct MAStop
    threshold::Union{AbstractFloat, Tuple}
    delta1::AbstractFloat
    delta2::AbstractFloat
    chi1::AbstractFloat
    chi2::AbstractFloat
end

function MAStop(;threshold=1e-10, delta1=0.9, delta2=1.1, chi1=0.01, chi2=0.01)
    return MAStop(threshold, delta1, delta2, chi1, chi2)
end


"""
    threshold_stop(log::MALoggerRecipe)

Function that takes an input threshold and stops when the most recent entry in the history
    vector is less than the threshold.

# Arguments
 - `log::MALoggerRecipe`, a structure containing the moving average logger information.

# Bool
 - Returns a Bool indicating if the stopping threshold is satisfied.
"""
# Common interface for stopping criteria
function check_stop_criterion(log::MALoggerRecipe)
    its = log.iterations
    if its > 0
        I_threshold = iota_threshold(log)
        thresholdChecks =
            sqrt(log.iota_hist[its]) <= I_threshold && log.resid_hist[its] <= log.threshold_info.threshold
    else
        thresholdChecks = false
    end
    return thresholdChecks
end


"""
    iota_threshold(log::MALoggerRecipe)

Function that computes the stopping criterion using the sub-Exponential distribution 
    from the `MALoggerRecipe`, and the stopping criterion information in TODO: use this stucture? 
    `MAStop`. This 
    function is not exported and thus not directly callable by the user.

# Arguments
- `log::MALoggerRecipe`, the log information for the moving average tracking.

# Ouputs
- Returns the stoppping criterion value.

Pritchard, Nathaniel, and Vivak Patel. "Solving, Tracking and Stopping Streaming Linear Inverse Problems." arXiv preprint arXiv:2201.05741 (2024).
"""
function iota_threshold(log::MALoggerRecipe)
    delta1 = log.threshold_info.delta1
    delta2 = log.threshold_info.delta2
    chi1 = log.threshold_info.chi1
    chi2 = log.threshold_info.chi2
    threshold = log.threshold_info.threshold
    lambda = log.ma_info.lambda
    # If the constants for the sub-Exponential distribution are not defined then define them

    if typeof(log.dist_info.sigma2) <: Nothing
        get_SE_constants!(log, log.dist_info.sampler)
    end
    #If there is an omega in the sub-Exponential distribution then skip that calculation 
    if typeof(log.dist_info.omega) <: Nothing
        # Compute the threshold bound in the case where there is no omega
        c = min(
            (1 - delta1)^2 * threshold^2 / (2 * log(1 / chi1)),
            (delta2 - 1)^2 * threshold^2 / (2 * log(1 / chi2)),
        )
        c /=
            (log.dist_info.sigma2 * sqrt(log.iota_log[log.iterations])) *
            (1 + log(lambda)) / lambda
    else
        #compute error bound when there is an omega
        siota =
            (log.dist_info.sigma2 * sqrt(log.iota_log[log.iterations])) *
            (1 + log(lambda)) / lambda
        min1 = min(
            (1 - delta1)^2 * threshold^2 / (2 * log(1 / chi1) * siota),
            lambda * (1 - delta1) * threshold / (2 * log(1 / chi1) * log.dist_info.omega),
        )
        min2 = min(
            (delta2 - 1)^2 * threshold^2 / (2 * log(1 / chi2) * siota),
            lambda * (delta2 - 1) * threshold / (2 * log(1 / chi2) * log.dist_info.omega),
        )
        c = min(min1, min2)
    end

    return c
end