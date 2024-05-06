# This file is part of RLinearAlgebra.jl

# using LinearAlgebra
"""
    MAInfo

A mutable structure that stores information relevant to the moving average of the progress estimator.

# Fields
- `lambda1::Int64`, the width of the moving average during the fast convergence phase of the algorithm,
  this has a default value of one.    
- `lambda2::Int64`, the width of the moving average in the slower convergence phase, this has a default value of 30.
- `lambda::Int64`, the value of the moving average width at the current iteration.
- `flag::Bool`, A boolean indicating which phase we are in, a value of `true` indicates second phase. 
- `idx::Int64`, The index indcating what value should be replaced in the moving average buffer.
- `res_window::Vector{Float64}` - The moving average buffer.
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
    DistInfo

A mutable structure that stores information about the sub-Exponential distribution.

# Fields
- `sampler::Union{DataType, Nothing}`, The type of sampler being used.
- `max_dimension::Int64`, The dimension that is being sampled.
- `block_dimension::Int64`, The sampling dimension.
- `sigma2::Union{Float64, Nothing}`, The variance parameter in the sub-Exponential distribution, 
   if not given is determined for sampling method.
- `omega::Union{Float64, Nothing}`, The exponential distrbution parameter, if not given is determined for sampling methods.
- `eta::Float64`, A parameter for adjusting the conservativeness of the distribution, higher value means a less conservative
  estimate. By default, this is set to one.
"""
mutable struct DistInfo
    sampler::Union{DataType, Nothing}
    max_dimension::Int64
    block_dimension::Int64
    sigma2::Union{Float64, Nothing}
    omega::Union{Float64, Nothing}
    eta::Float64
end

"""
    LSLogMA <: LinSysSolverLog

A mutable structure that stores information about a randomized linear solver's behavior.
    The log assumes that only a random block of full linear system is available for processing. 
    The goal of this log is usually for all use cases as it acts as a cheap approximation of the 
    full residual.

# Fields
- `collection_rate::Int64`, the frequency with which to record information to append to the
    remaining fields, starting with the initialization (i.e., iteration 0).
- `ma_info::MAInfo`, structure that holds information relevant only to moving average, 
   like the two choices of moving average widths (lambda1 and lambda2), the current moving average
   width (lambda), a flag to indicate which moving average regime, between lambda1 and lambda2, is 
   being used, a vector containing the values to be averaged (res_window), and an indicator of where 
   the next value should be placed (idx).
- `resid_hist::Vector{Float64}`, a structure that contains the moving average of the error proxy, 
   typically the norm of residual or gradient of normal equations.
- `iota_hist::Vector{Float64}`, a structure that contains the moving average of the error proxy, 
   typically the norm of residual or gradient of normal equations.
- `width_hist::Vector{Int64}`, data structure that contains the widths used in the moving average 
   calculation at each iteration.
- `resid_norm::Function`, a function that accepts a single vector argument and returns a
    scalar. Used to compute the residual size.
- `iterations::Int64`, the number of iterations of the solver.
- `converged::Bool`, a flag to indicate whether the system has converged by some measure.
- `true_res`, a boolean indicating if we want the true residual computed instead of approximate.
- `dist_info::DistInfo`, a structure that stores information about the sub-Exponential distribution.
# Constructors
- Calling `LSLogMA()` sets `collection_rate = 1`,  `lambda1 = 1`,
    `lambda2 = 30`, `resid_hist = Float64[]`, `iota_hist = Float64[]`, `width_hist = Int64[]`, 
    `resid_norm = norm` (Euclidean norm), `iterations = -1`, `converged = false`, 
    `sampler = LinSysVecRowDetermCyclic`, `max_dimension = 0`, `sigma2 = nothing`, `omega = nothing`,
    `eta = 1`, and `true_res = false`. The user can specify their own values of lambda1, lambda2, sigma2, omega, and eta using 
    key word arguments as inputs into the constructor.
"""
mutable struct LSLogMA <: LinSysSolverLog
    collection_rate::Int64
    ma_info::MAInfo
    resid_hist::Vector{Float64}
    iota_hist::Vector{Float64}
    width_hist::Vector{Int64}
    resid_norm::Function
    iterations::Int64
    converged::Bool
    true_res::Bool
    dist_info::DistInfo
end

LSLogMA(;
        cr = 1, 
        lambda1 = 1, 
        lambda2 = 30, 
        sigma2 = nothing, 
        omega = nothing, 
        eta = 1, 
        true_res = false
       ) = LSLogMA( cr,
                    MAInfo(1, lambda2, 1, false, 1, zeros(lambda2)),
                    Float64[], 
                    Float64[], 
                    Int64[],
                    norm, 
                    -1, 
                    false,
                    true_res,
                    DistInfo(nothing, 0, 0, sigma2, omega, eta) 
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
        if sum(supertype(typeof(sampler)) .<: [LinSysVecRowSampler, LinSysVecRowSketch, LinSysVecRowSelect]) > 1
            log.dist_info.max_dimension = size(A,1)
            log.dist_info.block_dimension = 1 
        elseif sum(supertype(typeof(sampler)) .<: [LinSysBlkRowSampler, LinSysBlkRowSketch, LinSysBlkRowSelect])
            log.dist_info.max_dimension = size(A,1)
            # For the block methods samp[3] is always the sketched residual, its length is block size
            log.dist_info.block_dimension = length(samp[3])
        elseif sum(supertype(typeof(sampler)) .<: [LinSysVecColSampler, LinSysVecColSketch, LinSysVecColSelect]) > 1
            log.dist_info.max_dimension = size(A,1)
            log.dist_info.block_dimension = 1 
        else
            log.dist_info.max_dimension = size(A,1)
            # For the block methods samp[3] is always the sketched residual, its length is block size
            log.dist_info.block_dimension = length(samp[3])
        end
        
        log.dist_info.sampler = typeof(sampler)  
    # If the constants for the sub-Exponential distribution are not defined then define them
        if typeof(log.dist_info.sigma2) <: Nothing
            get_SE_constants!(log, log.dist_info.sampler)
        end

    end

    ma_info = log.ma_info
    log.iterations = iter
    #Check if we want exact residuals computed
    if !log.true_res && iter > 0
        # Compute the current residual to second power to align with theory
        res::Float64 = eltype(samp[1]) <: Int64 || size(samp[1],2) != 1 ? 
            log.resid_norm(samp[3])^2 : log.resid_norm(dot(samp[1], x) - samp[2])^2 
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

Function that updates the moving average tracking statistic. This function is not exported and thus the user has no direct access. 

# Inputs
- `log::LSLogMA`, the moving average log structure.
- `res::Union{AbstractVector, Real}, the residual for the current iteration. This could be sketeched or full residual depending on the inputs when creating the log structor.
-`lambda_base::Int64`, which lambda, between lambda1 and lambda2, is currently being used.
-`iter::Int64`, the current iteration.

# Outputs
Performs and inplace update of the log datatype.
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
            push!(log.width_hist, ma_info.lambda)
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
            push!(log.width_hist, ma_info.lambda)
            push!(log.resid_hist, accum / ma_info.lambda) 
            push!(log.iota_hist, accum2 / ma_info.lambda) 
        end
        
        ma_info.lambda += 1
    end

end

#Function that will return rho and its uncertainty from a LSLogMA type 
"""
    get_uncertainty(log::LSLogMA; alpha = .95)
    
A function that takes a LSLogMA type and a confidence level, alpha, and returns credible intervals for for every rho in the log, specifically it returns a tuple with (rho, Upper bound, Lower bound).
"""
function get_uncertainty(hist::LSLogMA; alpha = .95)
    lambda = hist.ma_info.lambda
    l = length(hist.iota_hist)
    upper = zeros(l)
    lower = zeros(l)
    # If the constants for the sub-Exponential distribution are not defined then define them
    if typeof(hist.dist_info.sigma2) <: Nothing
        get_SE_constants!(hist, hist.dist_info.sampler)
    end
    
    for i in 1:l
        width = hist.width_hist[i]
        iota = hist.iota_hist[i]
        rho = hist.resid_hist[i]
        #Define the variance term for the Gaussian part
        cG = hist.dist_info.sigma2 * (1 + log(width)) * iota / (width * hist.dist_info.eta)
        #If there is an omega in the sub-Exponential distribution then skip that calculation 
        if typeof(hist.dist_info.omega) <: Nothing
            # Compute the threshold bound in the case where there is no omega
            diffG = sqrt(cG * 2 * log(2/(1-alpha)))
            upper[i] = rho + diffG
            lower[i] = rho - diffG
        else
            #compute error bound when there is an omega
            diffG = sqrt(cG * 2 * log(2/(1-alpha)))
            diffO = sqrt(iota) * 2 * log(2/(1-alpha)) * hist.dist_info.omega / (hist.dist_info.eta * width)
            diffM = min(diffG, diffO)
            upper[i] = rho + diffG
            lower[i] = rho - diffG
        end
        
    end

    return (hist.resid_hist, upper, lower)  

end

"""
    get_SE_constants!(log::LSLogMA, sampler::Type{T<:LinSysSampler})

A function that returns a default set of sub-Exponential constants for each sampling method. This function is not exported and thus the user does not have direct access to it. 

# Inputs 
- `log::LSLogMA`, the log containing al the tracking information.
- `sampler::Type{LinSysSampler}`, the type of sampler being used.

# Outputs
Performs an inplace update of the sub-Exponential constants for the log.
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
        end

    end

end


#Column subsetting methods have same constants as in row case
for type in (LinSysVecColOneRandCyclic, LinSysVecColDetermCyclic)
    @eval begin
        function get_SE_constants!(log::LSLogMA, sampler::Type{$type})
            log.dist_info.sigma2 = log.dist_info.max_dimension^2 / (4 * log.dist_info.block_dimension^2 * log.dist_info.eta)
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
