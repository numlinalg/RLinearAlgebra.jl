# This file is part of RLinearAlgebra.jl

"""
    LSStopMaxIterations <: LinSysStopCriterion

A structure that specifies a maximum iteration stopping criterion. That is, once a method
    achieves a certain number of iterations, it stops.

# Fields
- `max_iter::Int64`, the maximum number of iterations.
"""
struct LSStopMaxIterations <: LinSysStopCriterion
    max_iter::Int64
end

# Common interface for stopping criteria
function check_stop_criterion(
    log::LinSysSolverLog,
    stop::LSStopMaxIterations
)
    return log.iterations == stop.max_iter ? true : false
end
