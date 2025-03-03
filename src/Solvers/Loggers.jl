"""
    Logger

An abstract supertype for structures that contain user-controlled parameters for a logger, 
which has the goal of recording the progress of a linear solver and evaluating convergence.
"""
abstract type Logger end

"""
    LoggerRecipe

An abstract supertype for structures that contain user-controlled parameters and
preallocated memory for a logger, which has the goal of recording the progress of a linear 
solver and evaluating convergence.
"""
abstract type LoggerRecipe end

# Docstring Components
logger_arg_list = Dict{Symbol, String}(
    :logger => "`logger::Logger`, a user-specified logging method.",
    :logger_recipe => "`logger::LoggerRecipe`, a fully initialized realization for a 
    logging method for a specific linwar solver.",
    :A => "`A::AbstractMatrix`, a target matrix for compression.",
    :b => "`b::AbstractVector`, a possible target vector for compression.",
    :err => "`err::Float64`, an error value to be logged.",
    :iteration => "`iteration::Int64`, the iteration of the solver." 
)

logger_output_list = Dict{Symbol, String}(
    :logger_recipe => "A `LoggerRecipe` object."
)

logger_method_description = Dict{Symbol, String}(
    :complete_logger => "A function that generates a `LoggerRecipe` given the 
    arguments.",
    :update_logger => "A function that updates the `LoggerRecipe` in place given 
    arguments."
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
    return nothing
end

##############################
# Include Logger Files
###############################
