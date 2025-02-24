# This file is part of RLinearAlgebra.jl

"""
    LSLogMA <: LinSysSolverLog

A mutable structure that stores information for tracking a randomized linear solver's 
    progress.
    The log assumes that the entire linear system is **not** available for processing. 


# Fields
- `collection_rate::Int64`, the frequency with which to record information about progress estimators 
    to append to the remaining fields, starting with the initialization (i.e., iteration `0`). 
    For example, `collection_rate` = 3 means the iteration difference between each records is 3, 
    i.e. recording information at iteration `0`, `3`, `6`, `9`, ....
- `ma_info::MAInfo`, [`MAInfo`](@ref)
- `resid_hist::Vector{Float64}`, contains an estimate of the progress of the randomized
    solver. These values are stored at iterates specified by `collection_rate`.
- `iota_hist::Vector{Float64}`, contains an estimate used for calculating the variability
    of the progress estimators. These values are stored at iterates specified by
    `collection_rate`.
- `lambda_hist::Vector{Int64}`, contains the widths of the moving average.
   These values are stored at iterates specified by `collection_rate`.
- `resid_norm::Function`, the desired `norm` used. The default constructor sets this to the 
    Euclidean norm. 
- `iterations::Int64`, the current iteration of the solver.
- `converged::Bool`, a flag to indicate whether the system has converged by some measure. 
    By default this is `false`.
- `dist_info::SEDistInfo`, [`SEDistInfo`](@ref)

# Constructors
- The keyword constructor is defined as 
`LSLogMA(collection_rate = 1,
        lambda1 = 1, 
        lambda2 = 30, 
        resid_norm = norm #(Euclidean norm), 
        sigma2 = nothing, 
        omega = nothing,
        eta = 1, 
        )`

For more information see:
- Pritchard, Nathaniel, and Vivak Patel. "Solving, tracking and stopping streaming linear 
    inverse problems." Inverse Problems (2024). doi:10.1088/1361-6420/ad5583.
- Pritchard, Nathaniel, and Vivak Patel. “Towards Practical Large-Scale Randomized Iterative 
    Least Squares Solvers through Uncertainty Quantification.” SIAM/ASA J. Uncertainty 
    Quantification 11 (2022): 996-1024. doi.org/10.1137/22M1515057
"""
mutable struct LSLogMA <: LinSysSolverLog
    collection_rate::Int64
    ma_info::MAInfo
    resid_hist::Vector{Float64}
    iota_hist::Vector{Float64}
    lambda_hist::Vector{Int64}
    resid_norm::Function
    iterations::Int64
    converged::Bool
    dist_info::SEDistInfo
end

LSLogMA(;
        collection_rate = 1, 
        lambda1 = 1, 
        lambda2 = 30, 
        sigma2 = nothing, 
        omega = nothing, 
        eta = 1, 
       ) = LSLogMA( collection_rate,
                    MAInfo(lambda1, lambda2, lambda1, false, 1, zeros(lambda2)),
                    Float64[], 
                    Float64[], 
                    Int64[],
                    norm, 
                    -1, 
                    false,
                    SEDistInfo(nothing, 0, 0, sigma2, omega, eta, 0)
                  )

#Function to update the moving average
function log_update!(
    log::LSLogMA,
    sampler::LinSysSampler,
    x::AbstractVector,
    samp::Tuple,
    iter::Int64,
    A::AbstractArray,
    b::AbstractVector,
)
    if iter == 0 
        # Check if it is a row or column method and record dimensions
        log.dist_info.dimension = size(A,1)
        # For the block methods samp[3] is always the sketched residual, its length is block size
        log.dist_info.block_dimension = 
            if supertype(typeof(sampler)) <: LinSysVecRowSampler || supertype(typeof(sampler)) <: LinSysVecColSampler
                1
            elseif supertype(typeof(sampler)) <: LinSysBlkRowSampler || supertype(typeof(sampler)) <: LinSysBlkColSampler
                sampler.block_size
            else
                throw(ArgumentError("`sampler` is not of type `LinSysBlkColSampler`, `LinSysVecColSampler`, `LinSysBlkRowSampler`, or `LinSysVecRowSampler`"))
            end
        
        # If the constants for the sub-Exponential distribution are not defined then define them
        if typeof(log.dist_info.sigma2) <: Nothing || log.dist_info.sigma2 == 0
            get_SE_constants!(log, typeof(sampler))
        end
        
    end

    ma_info = log.ma_info
    log.iterations = iter
    # Compute the current residual to second power to align with theory
    # Check if it is one dimensional or block sampling method
    res::Float64 = log.dist_info.scaling * (eltype(samp[1]) <: Int64 || size(samp[1],2) != 1 ? 
                       log.resid_norm(samp[3])^2 : log.resid_norm(dot(samp[1], x) - samp[2])^2) 
  
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

