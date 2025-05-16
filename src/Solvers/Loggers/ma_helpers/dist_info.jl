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

function SEDistInfo(; sampler=nothing, dimension=0, block_dimension=0, sigma2=nothing, omega=nothing, eta=1.0, scaling=0.0)
    eta > 0 || throw(ArgumentError("eta must be positive"))
    new(sampler, dimension, block_dimension, sigma2, omega, eta, scaling)
end

#########################################
# Functions
#########################################
#Function that will return rho and its uncertainty from a LSLogMA type 
"""