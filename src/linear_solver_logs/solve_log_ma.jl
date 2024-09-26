# This file is part of RLinearAlgebra.jl
"""
    MAInfo

A mutable structure that stores information relevant to the moving average of the progress estimator.

# Fields
- `lambda1::Int64`, the width of the moving average during the fast convergence phase of the algorithm. 
  During this fast convergence phase, the majority of variation of the sketched estimator comes from 
  improvement in the solution and thus wide moving average windows inaccurately represent progress. 
- `lambda2::Int64`, the width of the moving average in the slower convergence phase. In the slow convergence
  phase, each iterate differs from the previous one by a small amount and thus most of the observed variation
  arises from the randomness of the sketched progress estimator, which is best smoothed by a wide moving
  average width.
- `lambda::Int64`, the width of the moving average at the current iteration. This value is not controlled by
  the user. 
- `flag::Bool`, a boolean indicating which phase we are in, a value of `true` indicates slow convergence phase. 
- `idx::Int64`, the index indcating what value should be replaced in the moving average buffer.
- `res_window::Vector{Float64}`, the moving average buffer.

For more information see:
- Pritchard, Nathaniel, and Vivak Patel. "Solving, tracking and stopping streaming linear 
    inverse problems." Inverse Problems (2024). doi:10.1088/1361-6420/ad5583.
- Pritchard, Nathaniel, and Vivak Patel. “Towards Practical Large-Scale Randomized Iterative 
    Least Squares Solvers through Uncertainty Quantification.” SIAM/ASA J. Uncertainty 
    Quantification 11 (2022): 996-1024. doi.org/10.1137/22M1515057
"""
mutable struct MAInfo
    lambda1::Int64
    lambda2::Int64
    lambda::Int64
    flag::Bool
    idx::Int64
    res_window::Vector{Float64}
end

""" 
    SEDistInfo

A mutable structure that stores information about a distribution (i.e., sampling method)
in the sub-Exponential family.

# Fields
- `sampler::Union{DataType, Nothing}`, the type of sampling method.
- `dimension::Int64`, the dimension that of the space that is being sampled.
- `block_dimension::Int64`, the dimension of the sample.
- `sigma2::Union{Float64, Nothing}`, the variance parameter in the sub-Exponential family. 
    If not specified by the user, a value is selected from a table based on the `sampler`. 
    If the `sampler` is not in the table, then `sigma2` is set to `1`.
- `omega::Union{Float64, Nothing}`, the exponential distrbution parameter. If not specified 
    by the user, a value is selected from a table based on the `sampler`.
    If the `sampler` is not in the table, then `omega` is set to `1`.
- `eta::Float64`, a parameter for adjusting the conservativeness of the distribution, higher 
    value means a less conservative estimate. A recommended value is `1`.
- `scaling::Float64`, a scaling parameter for the norm-squared of the sketched residual to 
    ensure its expectation is the norm-squared of the residual.

For more information see:
- Pritchard, Nathaniel, and Vivak Patel. "Solving, tracking and stopping streaming linear 
    inverse problems." Inverse Problems (2024). doi:10.1088/1361-6420/ad5583.
- Pritchard, Nathaniel, and Vivak Patel. “Towards Practical Large-Scale Randomized Iterative 
    Least Squares Solvers through Uncertainty Quantification.” SIAM/ASA J. Uncertainty 
    Quantification 11 (2022): 996-1024. doi.org/10.1137/22M1515057
"""
mutable struct SEDistInfo
    sampler::Union{DataType, Nothing}
    dimension::Int64
    block_dimension::Int64
    sigma2::Union{Float64, Nothing}
    omega::Union{Float64, Nothing}
    eta::Float64
    scaling::Float64
end

"""
    LSLogMA <: LinSysSolverLog

A mutable structure that stores information for tracking a randomized linear solver's 
    progress.
    The log assumes that the entire linear system is **not** available for processing. 


# Fields
- `collection_rate::Int64`, the frequency with which to record information about progress estimators 
    to append to the remaining fields, starting with the initialization (i.e., iterate `0`).
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
- `iteration::Int64`, the current iteration of the solver.
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
    iteration::Int64
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
        if supertype(typeof(sampler)) <: LinSysVecRowSampler
            log.dist_info.block_dimension = 1 
        elseif supertype(typeof(sampler)) <: LinSysBlkRowSampler
            # For the block methods samp[3] is always the sketched residual, its length is block size
            log.dist_info.block_dimension = sampler.block_size 
        elseif supertype(typeof(sampler)) <: LinSysVecColSampler
            log.dist_info.block_dimension = 1 
        elseif supertype(typeof(sampler)) <: LinSysBlkColSampler
            # For the block methods samp[3] is always the sketched residual, its length is block size
            log.dist_info.block_dimension = sampler.block_size
        else
            throw(ArgumentError("`sampler` is not of type `LinSysBlkColSampler`, `LinSysVecColSampler`, `LinSysBlkRowSampler`, or `LinSysVecRowSampler`"))
        end
        
        # If the constants for the sub-Exponential distribution are not defined then define them
        if typeof(log.dist_info.sigma2) <: Nothing || log.dist_info.sigma2 == 0
            get_SE_constants!(log, typeof(sampler))
        end
        
    end

    ma_info = log.ma_info
    log.iteration = iter
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

"""
    update_ma!(
        log::LSLogMA, 
        res::Union{AbstractVector, Real}, 
        lambda_base::Int64, 
        iter::Int64
    ) 

Function that updates the moving average tracking statistic. 

# Inputs
- `log::LSLogMA`, the moving average log structure.
- `res::Union{AbstractVector, Real}`, the sketched residual for the current iteration. 
- `lambda_base::Int64`, which lambda, between lambda1 and lambda2, is currently being used.
- `iter::Int64`, the current iteration.

# Outputs
Updates the log datatype and does not explicitly return anything.
"""
function update_ma!(log::LSLogMA, res::Union{AbstractVector, Real}, lambda_base::Int64, iter::Int64)
    # Variable to store the sum of the terms for rho
    accum = 0
    # Variable to store the sum of the terms for iota 
    accum2 = 0
    ma_info = log.ma_info
    ma_info.idx = ma_info.idx < ma_info.lambda2 && iter != 0 ? ma_info.idx + 1 : 1
    ma_info.res_window[ma_info.idx] = res
    #Check if entire storage buffer can be used
    if ma_info.lambda == ma_info.lambda2 
        # Compute the moving average
        for i in 1:ma_info.lambda2
            accum += ma_info.res_window[i]
            accum2 += ma_info.res_window[i]^2
        end
       
        if mod(iter, log.collection_rate) == 0 || iter == 0
            push!(log.lambda_hist, ma_info.lambda)
            push!(log.resid_hist, accum / ma_info.lambda) 
            push!(log.iota_hist, accum2 / ma_info.lambda) 
        end

    else
        # Consider the case when lambda <= lambda1 or  lambda1 < lambda < lambda2
        diff = ma_info.idx - ma_info.lambda
        # Because the storage of the residual is based dependent on lambda2 and 
        # we want to sum only the previous lamdda terms we could have a situation
        # where we want the first `idx` terms of the buffer and the last `diff`
        # terms of the buffer. Doing this requires two loops
        # If `diff` is negative there idx is not far enough into the buffer and
        # two sums will be needed
        startp1 = diff < 0 ? 1 : (diff + 1)
        
        # Assuming that the width of the buffer is lambda2 
        startp2 = diff < 0 ? ma_info.lambda2 + diff + 1 : 2 
        endp2 = diff < 0 ? ma_info.lambda2 : 1

        # Compute the moving average two loop setup required when lambda < lambda2
        for i in startp1:ma_info.idx
            accum += ma_info.res_window[i]
            accum2 += ma_info.res_window[i]^2
        end

        for i in startp2:endp2
            accum += ma_info.res_window[i]
            accum2 += ma_info.res_window[i]^2
        end

        #Update the log variable with the information for this update
        if mod(iter, log.collection_rate) == 0 || iter == 0
            push!(log.lambda_hist, ma_info.lambda)
            push!(log.resid_hist, accum / ma_info.lambda) 
            push!(log.iota_hist, accum2 / ma_info.lambda) 
        end
        
        ma_info.lambda += ma_info.lambda < lambda_base ? 1 : 0
    end

end

#Function that will return rho and its uncertainty from a LSLogMA type 
"""
    get_uncertainty(log::LSLogMA; alpha = 0.05)
    
A function that takes a LSLogMA type and a confidence level, `alpha`, and returns a `(1-alpha)`-credible intervals 
for every `rho` in the `log`, specifically it returns a tuple with (rho, Upper bound, Lower bound).
"""
function get_uncertainty(hist::LSLogMA; alpha::Float64 = 0.05)
    lambda = hist.ma_info.lambda
    l = length(hist.iota_hist)
    upper = zeros(l)
    lower = zeros(l)
    # If the constants for the sub-Exponential distribution are not defined then define them
    if typeof(hist.dist_info.sigma2) <: Nothing
        throw(ArgumentError("The SE constants are empty, please set them in dist_info field of LSLogMA first."))
    end
    
    for i in 1:l
        width = hist.lambda_hist[i]
        iota = hist.iota_hist[i]
        rho = hist.resid_hist[i]
        #Define the variance term for the Gaussian part
        cG = hist.dist_info.sigma2 * (1 + log(width)) * iota / (hist.dist_info.eta * width)
        #If there is an omega in the sub-Exponential distribution then skip that calculation 
        if typeof(hist.dist_info.omega) <: Nothing
            # Compute the threshold bound in the case where there is no omega
            diffG = sqrt(cG * 2 * log(2/(alpha)))
            upper[i] = rho + diffG
            lower[i] = rho - diffG
        else
            #compute error bound when there is an omega
            diffG = sqrt(cG * 2 * log(2/(alpha)))
            diffO = sqrt(iota) * 2 * log(2/(alpha)) * hist.dist_info.omega / (hist.dist_info.eta * width)
            diffM = max(diffG, diffO)
            upper[i] = rho + diffM
            lower[i] = rho - diffM
        end
        
    end

    return (hist.resid_hist, upper, lower)  
end

"""
    get_SE_constants!(log::LSLogMA, sampler::Type{T<:LinSysSampler})

A function that returns a default set of sub-Exponential constants for each sampling method. 
This function is not exported and thus the user does not have direct access to it. 

# Inputs 
- `log::LSLogMA`, the log containing all the tracking information.
- `sampler::Type{LinSysSampler}`, the type of sampler being used.

# Outputs
Performs an inplace update of the sub-Exponential constants for the log. Additionally, updates the scaling constant to ensure expectation of 
block norms is equal to true norm. If default is not a defined a warning is returned that sigma2 is set 1 and scaling is set to 1. 
"""
function get_SE_constants!(log::LSLogMA, sampler::Type{T}) where T<:LinSysSampler
        @warn "No constants defined for method of type $sampler. By default we set sigma2 to 1 and scaling to 1."
        log.dist_info.sigma2 = 1
        log.dist_info.scaling = 1 
end

for type in (LinSysVecRowDetermCyclic,LinSysVecRowHopRandCyclic,
             LinSysVecRowSVSampler, LinSysVecRowUnidSampler,
             LinSysVecRowOneRandCyclic, LinSysVecRowDistCyclic,
             LinSysVecRowResidCyclic, LinSysVecRowMaxResidual,
             LinSysVecRowRandCyclic,
             LinSysVecRowMaxDistance,)
    @eval begin
        function get_SE_constants!(log::LSLogMA, sampler::Type{$type})
            log.dist_info.sigma2 = log.dist_info.dimension^2 / (4 * log.dist_info.block_dimension^2 * log.dist_info.eta)
            log.dist_info.scaling = log.dist_info.dimension / log.dist_info.block_dimension
        end

    end

end


#Column subsetting methods have same constants as in row case
for type in (LinSysVecColOneRandCyclic, LinSysVecColDetermCyclic)
    @eval begin
        function get_SE_constants!(log::LSLogMA, sampler::Type{$type})
            log.dist_info.sigma2 = log.dist_info.dimension^2 / (4 * log.dist_info.block_dimension^2 * log.dist_info.eta)
            log.dist_info.scaling = log.dist_info.dimension / log.dist_info.block_dimension
        end

    end

end

# For row samplers with gaussian sampling we have sigma2 = 1/.2345 and omega = .1127
for type in (LinSysVecRowGaussSampler, LinSysVecRowSparseGaussSampler)
    @eval begin
        function get_SE_constants!(log::LSLogMA, sampler::Type{$type})
            log.dist_info.sigma2 = log.dist_info.block_dimension / (0.2345 * log.dist_info.eta)
            log.dist_info.omega = .1127
            log.dist_info.scaling = 1.
        end

    end

end
# Need to implement this for the uniform sampling