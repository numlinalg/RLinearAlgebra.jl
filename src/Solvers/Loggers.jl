"""
    Logger

An abstract supertype containing the user defined parameters for a logger, which has the 
goal of recording the progress of a linear solver and evaluating convergence.
"""
abstract type Logger end

"""
    LoggerRecipe

An abstract supertype containing all parameters for a particular logger this includes 
the user defined parameters in `Logger` and the necessary storage buffers,
"""
abstract type LoggerRecipe end

"""
    complete_logger(logger::Logger, A::AbstractMatrix, b::AbstractVector)

A function that combines the user-defined information contained in the `Logger`, the matrix
from the matrix `A`, and vector `b`. to produce a logger recipe.

### Arguments
- `logger::Logger`, the `Logger` data structure containing user defined parameters.
- `A::AbstractMatrix`, the matrix in the linear system.
- `b::AbstractVector`, the constant vector in the linear system.
"""
function complete_logger(logger::Logger, A::AbstractMatrix, b::AbstractVector)
    # By default the LoggerRecipe formed by applying the version of this function that only
    # requires the `Logger` and linear system.
    return complete_logger(logger, A)
end

"""
    update_logger!(logger::LoggerRecipe, err::Float64, iteration::Int64)

A function that updates the history and convergence information in the logger recipe.

### Arguments
- `loggger::LoggerRecipe`, the LoggerRecipe being updated.
- `err::Float64`, the value of the progress estimator.
- `iteration::Int64`, the iteration of the linear solver.

### Outputs
- Performs an inplace update to the history and convergence information contained in the 
LoggerRecipe.
"""
function update_logger!(logger::LoggerRecipe, err::Float64, iteration::Int64)
    return
end

##############################
# Include Logger Files
###############################
