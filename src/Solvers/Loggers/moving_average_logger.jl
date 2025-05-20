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
    collection_rate::Integer
    ma_info::MAInfo 
    resid_hist::Vector{AbstractFloat}
    lambda_hist::Vector{Integer}  
    resid_norm::Function
    iterations::Integer
    converged::Bool
end

FullMALogger(;
             collection_rate::Integer=1, 
             lambda1::Integer=1, 
             lambda2::Integer=30,
			 resid_norm::Function=norm, 
            ) = LSLogFullMA(collection_rate, 
                            MAInfo(lambda1, lambda2, lambda1, false, 1, zeros(lambda2)),
                            AbstractFloat[], 
                            Int64[],
                            resid_norm, 
                            -1, 
                            false
                           )


"""
    FullMALoggerRecipe <: LoggerRecipe

	TODO
This is a mutable struct that contains the `max_it` parameter and stores the error metric 
    in a vector. Checks convergence of the solver based on the log information.

# Fields

"""

















# Common interface for update
function update_logger!(
    log::FullMALogger,
    sampler::LinSysSampler,
    x::AbstractVector,
    samp::Tuple,
    iter::Int64,
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






