"""
	FullMALogger <: Logger

A structure that stores information of specification about a randomized linear solver's 
    behavior. The log assumes that the full linear system is available for processing. 
    The goal of this log is usually for research, development or testing as it is unlikely 
    that the entire residual vector is readily available.

# Fields
- `collection_rate::Integer`, the frequency with which to record information about progress 
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

	FullMALogger(;collection_rate=1, lambda1=1, lambda2=30, resid_norm=norm)

## Keywords
- `collection_rate::Integer`, the frequency with which to record information about progress 
	to append to the remaining fields, starting with the initialization 
	(i.e., iteration `0`). For example, `collection_rate` = 3 means the iteration 
	difference between each records is 3, i.e. recording information at 
	iteration `0`, `3`, `6`, `9`, .... By default, this is set to `1`.
- `lambda1::Integer`, the TODO. By default, this is set to `1`.
- `lambda2::Integer`, the TODO. By default, this is set to `30`.
- `resid_norm::Function`, a function that accepts a single vector argument and returns a
	scalar. Used to compute the residual size. By default, `norm`, which is Euclidean 
	norm, is set.

## Returns
- A `FullMALogger` object.

## Throws TODO
- `ArgumentError` if `compression_dim` is non-positive, if `nnz` is exceeds
    `compression_dim`, or if `nnz` is non-positive.
"""
struct FullMALogger <: Logger
    max_it::Integer
    collection_rate::Integer
    ma_info::MAInfo 
    threshold_info::MAStop
    stopping_criterion::Function
    # resid_hist::Vector{AbstractFloat}
    # lambda_hist::Vector{Integer}  
    # resid_norm::Function
    # iterations::Integer
    # converged::Bool
    function BasicLogger(max_it, collection_rate, threshold_info, stopping_criterion)
        if max_it < 0 
            throw(ArgumentError("Field `max_it` must be positive or 0."))
        elseif collection_rate < 1
            throw(ArgumentError("Field `colection_rate` must be positive."))
        elseif collection_rate > max_it && max_it > 0
            throw(ArgumentError("Field `colection_rate` must be smaller than `max_it`."))
        end

        return new(max_it, collection_rate, threshold_info, stopping_criterion)
    end

end

FullMALogger(;
             collection_rate::Integer=1, 
             lambda1::Integer=1, 
             lambda2::Integer=30,
			#  resid_norm::Function=norm, 
            ) = LSLogFullMA(collection_rate, 
                            MAInfo(lambda1, lambda2, lambda1, false, 1, zeros(lambda2)),
                            # AbstractFloat[], 
                            # Int64[],
                            # resid_norm, 
                            # -1, 
                            # false
                           )


"""
    FullMALoggerRecipe <: LoggerRecipe

	TODO
The recipe contains the information of `FullMALogger`, stores the error metric 
    in a vector. Checks convergence of the solver based on the log information.

# Fields

"""
mutable struct FullMALoggerRecipe{F<:Function} <: LoggerRecipe
    max_it::Integer
    error::AbstractFloat
    iteration::Integer
    record_location::Integer
    collection_rate::Integer
    converged::Bool
    resid_hist::Vector{AbstractFloat}
    lambda_hist::Vector{Integer}  
    threshold_info::MAStop
    stopping_criterion::F
end

function complete_logger(logger::FullMALogger)
    # By using ceil if we divide exactly we always have space to record last value, if it 
    # does not divide exactly we have one more than required and thus enough space to record
    # the last value
    max_collection = Int(ceil(logger.max_it / logger.collection_rate))
    # Use one more than max_it to collect
    res_hist = zeros(max_collection + 1)
    lambda_hist = zeros(max_collection + 1)
    return FullMALoggerRecipe{typeof(logger.stopping_criterion)}(logger.max_it,
                                                                0.0,
                                                                logger.threshold_info,
                                                                1,
                                                                1,
                                                                logger.collection_rate,
                                                                false,
                                                                res_hist,
                                                                lambda_hist,
                                                                MAStop,
                                                                logger.stopping_criterion
                                                               )
end

















# Common interface for update
function update_logger!(
    log::FullMALogger,
    x::AbstractVector,
    samp::Tuple,
    iter::Integer,
    A::AbstractArray,
    b::AbstractVector,
)
    # Update iteration counter
    log.iterations = iter

    ###############################
    # Implement moving average (MA)
    ###############################
    ma_info = log.ma_info 
    # Compute the current residual to second power to align with theory
    res_norm_iter =  log.resid_norm(A * x - b)
    res::Float64 = res_norm_iter^2

    # Check if MA is in lambda1 or lambda2 regime
    if ma_info.flag
        update_ma!(log, res, ma_info.lambda2, iter)
    else
        # Check if we can switch between lambda1 and lambda2 regime
        # If it is in the monotonic decreasing of the sketched residual then we are in a lambda1 regime
        # otherwise we switch to the lambda2 regime which is indicated by the changing of the flag
        # because update_ma changes res_window and ma_info.idx we must check condition first
        flag_cond = iter == 0 || res <= ma_info.res_window[ma_info.idx] 
        update_ma!(log, res, ma_info.lambda1, iter)
        ma_info.flag = !flag_cond
    end

end





