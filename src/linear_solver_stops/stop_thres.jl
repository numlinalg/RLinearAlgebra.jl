# This file is part of RLinearAlgebra.jl

"""
    LSStopThreshold <: LinSysStopCriterion

A structure that specifies a threshold for sufficient progress as a stopping critertion. That is, once a progress estimator 
    achieves a certain quality of solution, it stops.

# Fields
- `max_iter::Int64`, the maximum number of iterations.
- `threshold::Float64`, the value threshold for when sufficient progress has been achieved.
"""
struct LSStopThreshold <: LinSysStopCriterion
    max_iter::Int64
    thres::Float64
end

# Common interface for stopping criteria
function check_stop_criterion(
    log::LinSysSolverLog,
    stop::LSStopThreshold
)
    its = log.iteration
    thresholdCheck = its > 0 && log.resid_hist[its] < stop.thres ? true : false 
    return thresholdCheck || its == stop.max_iter ? true : false
end
