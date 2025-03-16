"""
    BasicLogger <: Logger

This is a mutable struct that contains the `max_it` parameter and stores the error metric in a vector.
Checks convergence of the solver based on the log information.

# Fields
- `max_it::Int64`, The maximum number of iterations for the solver.
- `threshold_info::Union{Float64, Tuple}`, The parameters used for stopping the algorithm.
- `collection_rate::Int64`, the rate that history is gathered.
- `stopping_criterion::Function`, function that evaluates the stopping criterion.
"""
struct BasicLogger <: Logger
    max_it::Int64
    collection_rate::Int64
    threshold_info::Union{Float64, Tuple}
    stopping_criterion::Function
end

BasicLogger(;
            max_it = 0, 
            collection_rate = 1, 
            threshold = 0.0,
            stopping_criterion = threshold_stop 
           ) = BasicLogger(max_it, collection_rate, threshold, stopping_criterion)

"""
    BasicLoggerRecipe <: LoggerRecipe

This is a mutable struct that contains the `max_it` parameter and stores the error metric in a vector.
Checks convergence of the solver based on the log information.

# Fields
- `max_it::Int64`, The maximum number of iterations for the solver.
- `error::Float64`, The current error metric.
- `threshold_info::Union{Float64, Tuple}`, The parameters used for stopping the algorithm.
- `iteration::Int64`, the current iteration of the solver.
- `record_location::Int64`, the location in the history vector of the most recent entry.
- `collection_rate::Int64`, the rate that history is gathered.
- `converged::Bool`, A boolean indicating whether the stopping criterion is satisfied.
- `StoppingCriterion::Function`, function that evaluates the stopping criterion.
- `hist:AbstractVector`, vector that contains the history of the error metric.
"""
mutable struct BasicLoggerRecipe{F<:Function} <: LoggerRecipe
    max_it::Int64
    error::Float64
    threshold_info::Union{Float64, Tuple}
    iteration::Int64
    record_location::Int64
    collection_rate::Int64
    converged::Bool
    stopping_criterion::F
    hist::Vector{Float64}
end


function complete_logger(logger::BasicLogger, A::AbstractMatrix)
    # We will run for a number of iterations equal to 3 itmes the number of rows if maxit is
    # not set
    max_it = logger.max_it == 0 ? 3 * size(A, 1) : logger.max_it

    max_collection = Int(ceil(max_it / logger.collection_rate))
    # use one more than max it form collection
    hist = zeros(max_collection + 1)
    return BasicLoggerRecipe{typeof(logger.stopping_criterion)}(max_it,
                                                                0.0,
                                                                logger.threshold_info,
                                                                1,
                                                                1,
                                                                logger.collection_rate,
                                                                false,
                                                                logger.stopping_criterion,
                                                                hist
                                                               )
end

function update_logger!(logger::BasicLoggerRecipe, error::Float64, iteration::Int64)
    logger.iteration = iteration
    logger.error = error
    if rem(iteration, logger.collection_rate) == 0
        logger.hist[logger.record_location] = error
        logger.record_location += 1
    end
    # Always check max_it stopping criterion
    # Compute in this way to avoid bounds error from searching in the max_it + 1 location
    logger.converged = iteration <= logger.max_it ? logger.stopping_criterion(logger) : false
    # If the algorithm has converged set the record location to 1 so the function  can be 
    # rerun
    if logger.converged || iteration == logger.max_it
        logger.record_location = 1
    end
    return

end

"""
    threshold_stop(log::BasicLoggerRecipe)

Function that takes an input threshold and stops when the most recent entry in the history
vector is less than the threshold.
"""
function threshold_stop(log::LoggerRecipe)
    return log.error < log.threshold_info
end
