"""
    Logger

An abstrct supertype for structures that record the progress of a `SolverRecipe` applied to
a coefficient matrix and constant vector.
"""
abstract type Logger end

"""
    LoggerRecipe

An abstract supertype for a structure that contains pre-allocated memory for a method that 
records the progress of a `SolverRecipe`. 
"""
abstract type LoggerRecipe end

# Docstring Components
logger_arg_list = Dict{Symbol,String}(
    :logger => "`logger::Logger`, a user-specified logging method.",
    :logger_recipe => "`logger::LoggerRecipe`, a fully initialized realization for a 
    logging method for a specific linear or least squares solver.",
    :A => "`A::AbstractMatrix`, a coefficient matrix.",
    :b => "`b::AbstractVector`, a constant vector.",
    :err => "`err::Float64`, an error value to be logged.",
    :iteration => "`iteration::Int64`, the iteration of the solver.",
)

logger_output_list = Dict{Symbol,String}(:logger_recipe => "A `LoggerRecipe` object.")

logger_method_description = Dict{Symbol,String}(
    :complete_logger => "A function that generates a `LoggerRecipe` given the 
    arguments.",
    :update_logger => "A function that updates the `LoggerRecipe` in place given 
    arguments.",
)
"""
    complete_logger(logger::Logger, A::AbstractMatrix, b::AbstractVector)

$(logger_method_description[:complete_logger])

### Arguments
- $(logger_arg_list[:logger])
- $(logger_arg_list[:A]) 
- $(logger_arg_list[:b]) 

### Outputs
- $(logger_output_list[:logger_recipe])
"""
function complete_logger(logger::Logger, A::AbstractMatrix, b::AbstractVector)
    # By default the LoggerRecipe formed by applying the version of this function that only
    # requires the `Logger` and linear system.
    return complete_logger(logger, A)
end

"""
    complete_logger(logger::Logger, A::AbstractMatrix)

$(logger_method_description[:complete_logger])

### Arguments
- $(logger_arg_list[:logger])
- $(logger_arg_list[:A]) 

### Outputs
- $(logger_output_list[:logger_recipe])
"""
function complete_logger(logger::Logger, A::AbstractMatrix)
    throw(
        ArgumentError("No `complete_logger` method defined for $(typeof(logger)) logger and
                      $(typeof(A)).")
    )
    return nothing 
end

"""
    update_logger!(logger::LoggerRecipe, err::Float64, iteration::Int64)

$(logger_method_description[:update_logger])

### Arguments
- $(logger_arg_list[:logger_recipe])
- $(logger_arg_list[:err]) 
- $(logger_arg_list[:iteration]) 

### Outputs
- Performs an inplace update to the `LoggerRecipe` and returns nothing.
"""
function update_logger!(logger::LoggerRecipe, err::Float64, iteration::Int64)
    throw(
        ArgumentError("No `update_logger!` method defined for $(typeof(logger)) 
                      `LoggerRecipe`, $(typeof(err)), and $(typeof(iteration)).")
    )
    return nothing
end

##############################
# Include Logger Files
###############################
