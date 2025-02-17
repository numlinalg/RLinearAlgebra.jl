# This file is part of RLinearAlgebra.jl

# This file contains the components that are needed for storing 
# and update moving average information for moving average method:
# Structs: MAInfo
# Functions: update_ma!

#########################################
# Structs
#########################################
"""
    MAInfo

A mutable structure that stores information relevant to the moving average of the 
progress estimator. This struct is used and updated in functions `update_ma`, 
and the `log_update!` of methods `solve_log_ma` and `solve_log_full_ma`.

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

#########################################
# Functions
#########################################
"""
    update_ma!(
        log::LinSysSolverLog, 
        res::Union{AbstractVector, Real}, 
        lambda_base::Int64, 
        iter::Int64
    ) 

Function that updates the moving average tracking statistic. 

# Inputs
- `log::LinSysSolverLog`, the parent structure of moving average log structure.
- `res::Union{AbstractVector, Real}`, the sketched residual for the current iteration. 
- `lambda_base::Int64`, which lambda, between lambda1 and lambda2, is currently being used.
- `iter::Int64`, the current iteration.

# Outputs
Updates the log datatype and does not explicitly return anything.
"""
function update_ma!(
    log::LinSysSolverLog,  # log::L where L <: LinSysSolverLog
    res::Union{AbstractVector, Real}, 
    lambda_base::Int64, 
    iter::Int64,
)
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
            (:iota_hist in fieldnames(typeof(log))) && push!(log.iota_hist, accum2 / ma_info.lambda) 
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
            (:iota_hist in fieldnames(typeof(log))) && push!(log.iota_hist, accum2 / ma_info.lambda) 
        end
        
        ma_info.lambda += ma_info.lambda < lambda_base ? 1 : 0
    end

end

