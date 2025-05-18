# This file contains the components that are needed for storing 
# and using distribution infromation for moving average method:
# Structs: SEDistInfo
# Functions: get_uncertainty, get_SE_constants!

#########################################
# Structs
#########################################
""" 
    SEDistInfo

A mutable structure that stores information about a distribution (i.e., sampling method)
in the sub-Exponential family.

# Fields
- `sampler::Union{DataType, Nothing}`, the type of sampling method.
- `dimension::Integer`, the dimension that of the space that is being sampled.
- `block_dimension::Integer`, the dimension of the sample.
- `sigma2::Union{AbstractFloat, Nothing}`, the variance parameter in the sub-Exponential family. 
    If not specified by the user, a value is selected from a table based on the `sampler`. 
    If the `sampler` is not in the table, then `sigma2` is set to `1`.
- `omega::Union{AbstractFloat, Nothing}`, the exponential distrbution parameter. If not specified 
    by the user, a value is selected from a table based on the `sampler`.
    If the `sampler` is not in the table, then `omega` is set to `1`.
- `eta::AbstractFloat`, a parameter for adjusting the conservativeness of the distribution, higher 
    value means a less conservative estimate. A recommended value is `1`.
- `scaling::AbstractFloat`, a scaling parameter for the norm-squared of the sketched residual to 
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
    dimension::Integer
    block_dimension::Integer
    sigma2::Union{AbstractFloat, Nothing}
    omega::Union{AbstractFloat, Nothing}
    eta::AbstractFloat
    scaling::AbstractFloat
    function SEDistInfo(sampler, dimension, block_dimension, sigma2, omega, eta, scaling)
        eta > 0 || throw(ArgumentError("eta must be positive"))
        return new(sampler, dimension, block_dimension, sigma2, omega, eta, scaling)
    end
end

SEDistInfo(;
            sampler=nothing, 
            dimension=0, 
            block_dimension=0, 
            sigma2=nothing, 
            omega=nothing, 
            eta=1.0, 
            scaling=0.0
           ) = SEDistInfo(sampler, dimension, block_dimension, sigma2, omega, eta, scaling)

#########################################
# Functions
#########################################
#Function that will return rho and its uncertainty from a LoggerRecipe type 
"""
    get_uncertainty(log::LoggerRecipe; alpha::AbstractFloat = 0.05)
    
A function that gets the uncertainty from LoggerRecipe or LSLogFullMA type.

# Arguments
- `hist::LoggerRecipe`, the parent structure of moving average log structure, 
    i.e. LoggerRecipe and LSLogFullMA types. Specifically, the information of 
    distribution (`dist_info`), and all histories stored in the structure.
- `alpha::AbstractFloat`, the confidence level. 

# Returns
- A `(1-alpha)`-credible intervals for every `rho` in the `log`, specifically 
    it returns a tuple with (rho, Upper bound, Lower bound).
"""
function get_uncertainty(hist::LoggerRecipe; alpha::AbstractFloat = 0.05)
    l = length(hist.iota_hist)
    upper = zeros(l)
    lower = zeros(l)
    # If the constants for the sub-Exponential distribution are not defined then define them
    if typeof(hist.dist_info.sigma2) <: Nothing
        throw(ArgumentError("The SE constants are empty, please set them in dist_info field of LoggerRecipe first."))
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
    get_SE_constants!(log::LoggerRecipe, sampler::Type{T<:CompressorRecipe})

A function that returns a default set of sub-Exponential constants for each sampling method. 
    This function is not exported and thus the user does not have direct access to it. 

# Arguments 
- `log::LoggerRecipe`, the log containing all the tracking information. Specifically, 
    the information of distribution (`dist_info`).
- `sampler::Type{CompressorRecipe}`, the type of sampler being used.

# Returns
- Performs an inplace update of the sub-Exponential constants for the log. Additionally, 
    updates the scaling constant to ensure expectation of block norms is equal to true norm. 
    If default is not a defined a warning is returned that sigma2 is set 1 and scaling 
    is set to 1. 
"""
function get_SE_constants!(log::LoggerRecipe, sampler::Type{T}) where T<:CompressorRecipe
        @warn "No constants defined for method of type $sampler. By default we set sigma2 to 1 and scaling to 1."
        log.dist_info.sigma2 = 1
        log.dist_info.scaling = 1 
end

for type in (LinSysVecRowDetermCyclic,LinSysVecRowHopRandCyclic,
             LinSysVecRowOneRandCyclic, LinSysVecRowSVSampler, 
             LinSysVecRowRandCyclic, LinSysVecRowUnidSampler, 
             LinSysVecRowDistCyclic, LinSysVecRowResidCyclic, 
             LinSysVecRowMaxResidual, LinSysVecRowMaxDistance,)
    @eval begin
        function get_SE_constants!(log::LoggerRecipe, sampler::Type{$type})
            log.dist_info.sigma2 = log.dist_info.dimension^2 / (4 * log.dist_info.block_dimension^2 * log.dist_info.eta)
            log.dist_info.scaling = log.dist_info.dimension / log.dist_info.block_dimension
        end

    end

end


# Column subsetting methods have same constants as in row case
for type in (LinSysVecColOneRandCyclic, LinSysVecColDetermCyclic)
    @eval begin
        function get_SE_constants!(log::LoggerRecipe, sampler::Type{$type})
            log.dist_info.sigma2 = log.dist_info.dimension^2 / (4 * log.dist_info.block_dimension^2 * log.dist_info.eta)
            log.dist_info.scaling = log.dist_info.dimension / log.dist_info.block_dimension
        end

    end

end

# For row samplers with gaussian sampling we have sigma2 = 1/.2345 and omega = .1127
for type in (LinSysVecRowGaussSampler, LinSysVecRowSparseGaussSampler)
    @eval begin
        function get_SE_constants!(log::LoggerRecipe, sampler::Type{$type})
            log.dist_info.sigma2 = log.dist_info.block_dimension / (0.2345 * log.dist_info.eta)
            log.dist_info.omega = .1127
            log.dist_info.scaling = 1.
        end

    end

end