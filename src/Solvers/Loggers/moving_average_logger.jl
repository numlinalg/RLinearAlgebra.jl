"""
	MALogger <: Logger

A structure that stores information of specification about a randomized linear solver's 
    behavior. The log assumes that the full linear system is available for processing. 
    The goal of this log is usually for research, development or testing as it is unlikely 
    that the entire residual vector is readily available.

# Fields TODO
- `collection_rate::Int64`, the frequency with which to record information about progress 
	to append to the remaining fields, starting with the initialization 
	(i.e., iteration `0`). For example, `collection_rate` = 3 means the iteration 
	difference between each records is 3, i.e. recording information at 
	iteration `0`, `3`, `6`, `9`, ....
- `ma_info::MAInfo`, [`MAInfo`](@ref)
- `resid_hist::Vector{AbstractFloat}`, retains a vector of numbers corresponding to the residual
	(uses the whole system to compute the residual). These values are stored at iterates 
	specified by `collection_rate`.
- `lambda_hist::Vector{Integer}`, contains the widths of the moving average.
	These values are stored at iterates specified by `collection_rate`.
- `resid_norm::Function`, a function that accepts a single vector argument and returns a
	scalar. Used to compute the residual size.
- `iterations::Integer`, the number of iterations of the solver.
- `converged::Bool`, a flag to indicate whether the system has converged by some measure. 
	By default this is `false`.

# Constructors

	MALogger(;collection_rate=1, lambda1=1, lambda2=30)

## Keywords
- `collection_rate::Integer`, the frequency with which to record information about progress 
	to append to the remaining fields, starting with the initialization 
	(i.e., iteration `0`). For example, `collection_rate` = 3 means the iteration 
	difference between each records is 3, i.e. recording information at 
	iteration `0`, `3`, `6`, `9`, .... By default, this is set to `1`.
- `lambda1::Integer`, the TODO. By default, this is set to `1`.
- `lambda2::Integer`, the TODO. By default, this is set to `30`.

## Returns
- A `MALogger` object.

## Throws TODO
- `ArgumentError` if `compression_dim` is non-positive, if `nnz` is exceeds
    `compression_dim`, or if `nnz` is non-positive.
"""
struct MALogger <: Logger
    max_it::Int64
    collection_rate::Integer
    ma_info::MAInfo 
    threshold_info::Union{Float64, Tuple}
    stopping_criterion::Function
    function MALogger(max_it, collection_rate, ma_info, threshold_info, stopping_criterion)
        if max_it < 0 
            throw(ArgumentError("Field `max_it` must be positive or 0."))
        elseif collection_rate < 1
            throw(ArgumentError("Field `colection_rate` must be positive."))
        elseif collection_rate > max_it && max_it > 0
            throw(ArgumentError("Field `colection_rate` must be smaller than `max_it`."))
        end

        return new(max_it, collection_rate, ma_info, threshold_info, stopping_criterion)
    end

end

MALogger(;
         max_it=0,
         collection_rate=1, 
         lambda1=1, 
         lambda2=30,
         threshold_info=1e-10,
         stopping_criterion=threshold_stop
        ) = MALogger(max_it,
                     collection_rate, 
                     MAInfo(lambda1, lambda2, lambda1, false, 1, zeros(lambda2)),
                     threshold_info,
                     stopping_criterion
                    )


"""
    MALoggerRecipe <: LoggerRecipe

	TODO
The recipe contains the information of `MALogger`, stores the error metric 
    in a vector. Checks convergence of the solver based on the log information.

# Fields

"""
mutable struct MALoggerRecipe{F<:Function} <: LoggerRecipe
    max_it::Int64
    error::AbstractFloat
    iteration::Int64
    record_location::Int64
    collection_rate::Integer
    converged::Bool
    ma_info::MAInfo
    resid_hist::Vector{AbstractFloat}
    lambda_hist::Vector{Integer}  
    threshold_info::Union{Float64, Tuple}
    stopping_criterion::F
end

function complete_logger(logger::MALogger)
    # By using ceil if we divide exactly we always have space to record last value, if it 
    # does not divide exactly we have one more than required and thus enough space to record
    # the last value
    max_collection = Int(ceil(logger.max_it / logger.collection_rate))
    # Use one more than max_it to collect
    res_hist = zeros(max_collection + 1)
    lambda_hist = zeros(max_collection + 1)
    return MALoggerRecipe{typeof(logger.stopping_criterion)}(logger.max_it,
                                                             0.0,
                                                             1,
                                                             1,
                                                             logger.collection_rate,
                                                             false,
                                                             logger.ma_info,
                                                             res_hist,
                                                             lambda_hist,
                                                             logger.threshold_info,
                                                             logger.stopping_criterion
                                                            )
end


# Common interface for update
function update_logger!(
    logger::MALoggerRecipe,
    error::AbstractFloat,
    iteration::Int64
)
    # Update iteration counter
    logger.iteration = iteration
    logger.error = error

    # Always check max_it stopping criterion
    # Compute in this way to avoid bounds error from searching in the max_it + 1 location
    logger.converged = iteration <= logger.max_it ? 
        logger.stopping_criterion(logger, logger.threshold_info) : 
        true 
    
    # log according to collection rate or if we have converged 
    if rem(iteration, logger.collection_rate) == 0 || logger.converged 
        ###############################
        # Implement moving average (MA)
        ###############################
        ma_info = logger.ma_info 
        # Compute the current residual to second power to align with theory
        res::AbstractFloat = error^2

        # Check if MA is in lambda1 or lambda2 regime
        if ma_info.flag
            update_ma!(logger, res, ma_info.lambda2, iteration)
        else
            # Check if we can switch between lambda1 and lambda2 regime
            # If it is in the monotonic decreasing of the sketched residual then we are in a lambda1 regime
            # otherwise we switch to the lambda2 regime which is indicated by the changing of the flag
            # because update_ma changes res_window and ma_info.idx we must check condition first
            flag_cond = iteration == 0 || res <= ma_info.res_window[ma_info.idx] 
            update_ma!(logger, res, ma_info.lambda1, iteration)
            ma_info.flag = !flag_cond
        end
        logger.record_location += 1
    end

    return nothing

end



function reset_logger!(logger::MALoggerRecipe)
    logger.error = 0.0
    logger.iteration = 1
    logger.record_location = 1
    logger.converged = false
    fill!(logger.resid_hist, 0.0)
    fill!(logger.lambda_hist, 0.0)
    return nothing
end



"""
    threshold_stop(log::MALoggerRecipe)

Function that takes an input threshold and stops when the most recent entry in the history
vector is less than the threshold.

# Arguments
 - `log::MALoggerRecipe`, a structure containing the logger information

# Bool
 - Returns a Bool indicating if the stopping threshold is satisfied.
"""
function threshold_stop(log::MALoggerRecipe)
    return log.error < log.threshold_info
end
