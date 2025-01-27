# This file is part of RLinearAlgebra.jl

"""
    LSLogFullMA <: LinSysSolverLog

A mutable structure that stores information about a randomized linear solver's behavior.
    The log assumes that the full linear system is available for processing. The goal of
    this log is usually for research, development or testing as it is unlikely that the
    entire residual vector is readily available.

# Fields
- `collection_rate::Int64`, the frequency with which to record information about progress 
    to append to the remaining fields, starting with the initialization 
    (i.e., iteration `0`). For example, `collection_rate` = 3 means the iteration 
    difference between each records is 3, i.e. recording information at 
    iteration `0`, `3`, `6`, `9`, ....
- `ma_info::MAInfo`, [`MAInfo`](@ref)
- `resid_hist::Vector{Float64}`, retains a vector of numbers corresponding to the residual
    (uses the whole system to compute the residual). These values are stored at iterates 
    specified by `collection_rate`.
- `lambda_hist::Vector{Int64}`, contains the widths of the moving average.
    These values are stored at iterates specified by `collection_rate`.
- `resid_norm::Function`, a function that accepts a single vector argument and returns a
    scalar. Used to compute the residual size.
- `iterations::Int64`, the number of iterations of the solver.
- `converged::Bool`, a flag to indicate whether the system has converged by some measure. 
    By default this is `false`.

# Constructors
- The keyword constructor is defined as 
`LSLogFullMA(collection_rate = 1,
            lambda1 = 1, 
            lambda2 = 30, 
            resid_norm = norm #(Euclidean norm), 
            )`
"""
mutable struct LSLogFullMA <: LinSysSolverLog
    collection_rate::Int64
    ma_info::MAInfo 
    resid_hist::Vector{Float64}
    lambda_hist::Vector{Int64}  
    resid_norm::Function
    iterations::Int64
    converged::Bool
end

LSLogFullMA(;
            collection_rate = 1, 
            lambda1 = 1, 
            lambda2 = 30, 
           ) = LSLogFullMA( collection_rate, 
                            MAInfo(lambda1, lambda2, lambda1, false, 1, zeros(lambda2)),
                            Float64[], 
                            Int64[],
                            norm, 
                            -1, 
                            false
                          )

# Common interface for update
function log_update!(
    log::LSLogFullMA,
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


