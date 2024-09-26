# This file is part of RLinearAlgebra.jl
"""
    MAInfo

A mutable structure that stores information relevant to the moving average of the progress estimator.

# Fields
- `lambda1::Int64`, the width of the moving average during the fast convergence phase of the algorithm. 
  During this fast convergence phase, the majority of variation of the sketched estimator comes from 
  improvement in the solution and thus wide moving average windows inaccurately represent progress. 
  By default this parameter is set to `1`.
- `lambda2::Int64`, the width of the moving average in the slower convergence phase. In the slow convergence
  phase, each iterate differs from the previous one by a small amount and thus most of the observed variation
  arises from the randomness of the sketched progress estimator, which is best smoothed by a wide moving
  average width. By default this parameter is set to `30`.
- `lambda::Int64`, the width of the moving average at the current iteration. This value is not controlled by
  the user. 
- `flag::Bool`, a boolean indicating which phase we are in, a value of `true` indicates second phase. 
- `idx::Int64`, the index indcating what value should be replaced in the moving average buffer.
- `res_window::Vector{Float64}`, the moving average buffer.

For more information see:
 - Pritchard, Nathaniel, and Patel, Vivak. Solving, tracking and stopping streaming linear inverse problems. 
Inverse Problems 40.8 Web. doi:10.1088/1361-6420/ad5583.
 - Pritchard, Nathaniel and Vivak Patel. “Towards Practical Large-Scale Randomized Iterative Least Squares 
Solvers through Uncertainty Quantification.” SIAM/ASA J. Uncertain. Quantification 11 (2022): 996-1024.
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

A mutable structure that stores information about the sub-Exponential family.

# Fields
<<<<<<< HEAD
- `sampler::Union{DataType, Nothing}`, the type of sampling method.
- `dimension::Int64`, the dimension that of the space that is being sampled.
- `block_dimension::Int64`, the dimension of the sample.
- `sigma2::Union{Float64, Nothing}`, the variance parameter in the sub-Exponential family, 
   if not specified by the user it is provided based on the sampling method.
- `omega::Union{Float64, Nothing}`, the exponential distrbution parameter, if not specified by the user, 
   it is provided based on the sampling method.
- `eta::Float64`, a parameter for adjusting the conservativeness of the distribution, higher value means a less conservative
  estimate. By default, this is set to `1`.

For more information see:
- Pritchard, Nathaniel, and Vivak Patel. "Solving, tracking and stopping streaming linear inverse problems." Inverse Problems (2024). doi:10.1088/1361-6420/ad5583.
- Pritchard, Nathaniel and Vivak Patel. “Towards Practical Large-Scale Randomized Iterative Least Squares 
Solvers through Uncertainty Quantification.” SIAM/ASA J. Uncertain. Quantification 11 (2022): 996-1024. doi.org/10.1137/22M1515057 
=======
- `sampler::Union{DataType, Nothing}`, The type of sampler being used.
- `max_dimension::Int64`, The dimension that is being sampled.
- `block_dimension::Int64`, The sampling dimension.
- `sigma2::Union{Float64, Nothing}`, The variance parameter in the sub-Exponential distribution, 
   if not given is determined for sampling method.
- `omega::Union{Float64, Nothing}`, The exponential distrbution parameter, if not given is determined for sampling methods.
- `eta::Float64`, A parameter for adjusting the conservativeness of the distribution, higher value means a less conservative
  estimate. By default, this is set to one.
- `scaling::Float64`, constant multiplied by norm to ensure expectation of block norms is the same as the full norm.
>>>>>>> 44a288b (Corrected Scaling issue so now sketched block expectation is same as true block)
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

A mutable structure that stores information about a randomized linear solver's behavior.
The log assumes that only a random block of full linear system is available for processing. 
The goal of this log is usually for all use cases as it acts as a cheap approximation of the 
full residual.

For more information see:
- Pritchard, Nathaniel, and Vivak Patel. "Solving, tracking and stopping streaming linear inverse problems." Inverse Problems (2024). doi:10.1088/1361-6420/ad5583.
- Pritchard, Nathaniel and Vivak Patel. “Towards Practical Large-Scale Randomized Iterative Least Squares 
Solvers through Uncertainty Quantification.” SIAM/ASA J. Uncertain. Quantification 11 (2022): 996-1024. doi.org/10.1137/22M1515057 

# Fields
- `collection_rate::Int64`, the frequency with which to record information about progress estimators 
    to append to the remaining fields, starting with the initialization (i.e., iterate `0`).
- `ma_info::MAInfo`, [`MAInfo`](@ref)
- `resid_hist::Vector{Float64}`, a structure that contains the moving average of the error proxy squared, 
   typically the norm of residual or gradient of normal equations, it is collected at a rate 
   specified by `collection_rate`.
- `iota_hist::Vector{Float64}`, a structure that contains the moving average of the error proxy
   to the fourth power, typically the norm of residual or gradient of normal equations. This is used 
   in part to approximate the variance of the estimator, it is collected at a rate specified by `collection_rate`.
- `lambda_hist::Vector{Int64}`, data structure that contains the lambdas, widths of the moving average,
   calculation, it is collected at a rate specified by `collection_rate`.
- `resid_norm::Function`, a function that accepts a single vector argument and returns a
    scalar. Used to compute the residual size.
- `iterations::Int64`, the current iteration of the solver.
- `converged::Bool`, a flag to indicate whether the system has converged by some measure. By default this
  is set to false.
- `true_res::Bool`, a boolean indicating if we want the true residual computed instead of approximate.
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
        true_res = false)`
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
    true_res::Bool
    dist_info::SEDistInfo
end

LSLogMA(;
        collection_rate = 1, 
        lambda1 = 1, 
        lambda2 = 30, 
        sigma2 = nothing, 
        omega = nothing, 
        eta = 1, 
        true_res = false
       ) = LSLogMA( collection_rate,
                    MAInfo(lambda1, lambda2, 1, false, 1, zeros(lambda2)),
                    Float64[], 
                    Float64[], 
                    Int64[],
                    norm, 
                    -1, 
                    false,
                    true_res,
                    DistInfo(nothing, 0, 0, sigma2, omega, eta, 0) 
                  )

#Function to update the moving average
function log_update!(
    log::LSLogMA,
    sampler::LinSysSampler,
    x::AbstractVector,
    samp::Tuple,
    iter::Int64,
    A::AbstractArray,
    b::AbstractVector
)
    if iter == 0 
        # Check if it is a row or column method and record dimensions
        if supertype(typeof(sampler)) <: LinSysVecRowSampler
            log.dist_info.dimension = size(A,1)
            log.dist_info.block_dimension = 1 
        elseif supertype(typeof(sampler)) <: LinSysBlkRowSampler
            log.dist_info.dimension = size(A,1)
            # For the block methods samp[3] is always the sketched residual, its length is block size
            log.dist_info.block_dimension = sampler.block_size 
        elseif supertype(typeof(sampler)) <: LinSysVecColSampler
            log.dist_info.dimension = size(A,1)
            log.dist_info.block_dimension = 1 
        else
            log.dist_info.dimension = size(A,1)
            # For the block methods samp[3] is always the sketched residual, its length is block size
            log.dist_info.block_dimension = sampler.block_size
        end
        
        log.dist_info.sampler = typeof(sampler)  
    # If the constants for the sub-Exponential distribution are not defined then define them
        if typeof(log.dist_info.sigma2) <: Nothing || log.dist_info.sigma2 == 0
            get_SE_constants!(log, log.dist_info.sampler)
        end

    end

    ma_info = log.ma_info
    log.iterations = iter
    #Check if we want exact residuals computed
    if !log.true_res && iter > 0
        # Compute the current residual to second power to align with theory
        res::Float64 = log.dist_info.scaling *
            (eltype(samp[1]) <: Int64 || size(samp[1],2) != 1 ? 
                                log.resid_norm(samp[3])^2 : 
                                log.resid_norm(dot(samp[1], x) - samp[2])^2) 
    else 
        res = log.resid_norm(A * x - b)^2 
    end
    # Check if MA is in lambda1 or lambda2 regime
    if ma_info.flag
        update_ma!(log, res, ma_info.lambda2, iter)
    else
        #Check if we can switch between lambda1 and lambda2 regime
        if res < ma_info.res_window[ma_info.idx]
            update_ma!(log, res, ma_info.lambda1, iter)
        else
            update_ma!(log, res, ma_info.lambda1, iter)
            ma_info.flag = true 
        end

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
- `res::Union{AbstractVector, Real}, the residual for the current iteration. 
    This could be sketeched or full residual depending on the inputs when creating the log structor.
- `lambda_base::Int64`, which lambda, between lambda1 and lambda2, is currently being used.
- `iter::Int64`, the current iteration.

# Outputs
Updates the log datatype and does not explicitly return anything.
"""
function update_ma!(log::LSLogMA, res::Union{AbstractVector, Real}, lambda_base::Int64, iter::Int64)
    accum = 0
    accum2 = 0
    ma_info = log.ma_info
    ma_info.idx = ma_info.idx < ma_info.lambda2 ? ma_info.idx + 1 : 1
    ma_info.res_window[ma_info.idx] = res
    #Check if entire storage buffer can be used
    if ma_info.lambda == lambda_base 
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
        # Get the difference between the start and current lambda
        diff = ma_info.idx - ma_info.lambda
        
        # Determine start point for first loop
        startp1 = diff < 0 ? 1 : (diff + 1)
        
        # Determine start and endpoints for second loop
        startp2 = diff > 0 ? 2 : lambda_base + diff + 1  
        endp2 = diff > 0 ? 1 : lambda_base

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
        
        ma_info.lambda += 1
    end

end

#Function that will return rho and its uncertainty from a LSLogMA type 
"""
    get_uncertainty(log::LSLogMA; alpha = 0.05)
    
A function that takes a LSLogMA type and a confidence level, `alpha`, and returns a `(1-alpha)`-credible intervals for for every rho in the log, specifically it returns a tuple with (rho, Upper bound, Lower bound).
"""
function get_uncertainty(hist::LSLogMA; alpha::Float64 = 0.05)
    lambda = hist.ma_info.lambda
    l = length(hist.iota_hist)
    upper = zeros(l)
    lower = zeros(l)
    # If the constants for the sub-Exponential distribution are not defined then define them
    if typeof(hist.dist_info.sigma2) <: Nothing
        get_SE_constants!(hist, hist.dist_info.sampler)
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
            diffM = min(diffG, diffO)
            upper[i] = rho + diffG
            lower[i] = rho - diffG
        end
        
    end

    return (hist.resid_hist, upper, lower)  
end

"""
    get_SE_constants!(log::LSLogMA, sampler::Type{T<:LinSysSampler})

A function that returns a default set of sub-Exponential constants for each sampling method. 
This function is not exported and thus the user does not have direct access to it. 

# Inputs 
- `log::LSLogMA`, the log containing al the tracking information.
- `sampler::Type{LinSysSampler}`, the type of sampler being used.

# Outputs
Performs an inplace update of the sub-Exponential constants for the log. Additionally, updates the scaling constant to ensure expectation of 
block norms is equal to true norm.
"""
function get_SE_constants!(log::LSLogMA, sampler::Type{T}) where T<:LinSysSampler
    return nothing
end

for type in (LinSysVecRowDetermCyclic,LinSysVecRowHopRandCyclic,
             LinSysVecRowSVSampler, LinSysVecRowUnidSampler,
             LinSysVecRowOneRandCyclic, LinSysVecRowDistCyclic,
             LinSysVecRowResidCyclic, LinSysVecRowMaxResidual,
             LinSysVecRowRandCyclic,
             LinSysVecRowMaxDistance,)
    @eval begin
        function get_SE_constants!(log::LSLogMA, sampler::Type{$type})
            log.dist_info.sigma2 = log.dist_info.max_dimension^2 / (4 * log.dist_info.block_dimension^2 * log.dist_info.eta)
            log.dist_info.scaling = log.dist_info.max_dimension / log.dist_info.block_dimension
        end

    end

end


#Column subsetting methods have same constants as in row case
for type in (LinSysVecColOneRandCyclic, LinSysVecColDetermCyclic)
    @eval begin
        function get_SE_constants!(log::LSLogMA, sampler::Type{$type})
            log.dist_info.sigma2 = log.dist_info.max_dimension^2 / (4 * log.dist_info.block_dimension^2 * log.dist_info.eta)
            log.dist_info.scaling = log.dist_info.max_dimension / log.dist_info.block_dimension
        end

    end

end

# For row samplers with gaussian sampling we have sigma2 = 1/.2345 and omega = .1127
for type in (LinSysVecRowGaussSampler, LinSysVecRowSparseGaussSampler)
    @eval begin
        function get_SE_constants!(log::LSLogMA, sampler::Type{$type})
            log.dist_info.sigma2 = log.dist_info.block_dimension / (0.2345 * log.dist_info.eta)
            log.dist_info.omega = .1127
        end

    end

end
# Need to implement this for the uniform sampling
