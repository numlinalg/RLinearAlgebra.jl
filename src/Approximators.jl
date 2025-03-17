"""
    Approximator

An abstract supertype for structures that store user-controlled parameters corresponding to
techniques that form low-rank approximations of the matrix `A`.
"""
abstract type Approximator end

"""
    ApproximatorRecipe

An abstract supertype for structures that store user-controlled parameters, linear system
dependent parameters and preallocated memory corresponding to techniques that form low-rank
approximations of the matrix `A`.
"""
abstract type ApproximatorRecipe end

"""
    ApproximatorError

An abstract supertype for structures containing user-controlled parameters corresponding to
methods that evaluate the quality of a low-rank approximation of a matrix `A`.
"""
abstract type ApproximatorError end

"""
    ApproximatorErrorRecipe

An abstract supertype for structures containing user-controlled parameters, matrix
dependent parameters and preallocated memory corresponding to methods that evaluate the
quality of a low-rank approximation of a matrix `A`.
"""
abstract type ApproximatorErrorRecipe end

# Docstring Components
approx_arg_list = Dict{Symbol,String}(
    :approximator => "`approximator::Approximator`, a data structure containing the
    user-defined parameters associated with a particular low-rank approximation.",
    :approximator_recipe => "`approximator::ApproximatorRecipe`, a fully initialized
    realization for a low rank approximation method for a particular matrix.",
    :approximator_error => "`error::ApproximatorError`, a data structure containing
    the user-defined parameters associated with a particular low-rank approximation error
    method.",
    :approximator_error_recipe => "`error::ApproximatorErrorRecipe`, a fully initialized
    realization for a low rank approximation error method for a particular matrix.",
    :A => "`A::AbstractMatrix`, a target matrix for approximation.",
    :compressor_recipe => "`S::CompressorRecipe`, a fully initialized realization for a 
    compression method for a specific matrix or collection of matrices and vectors.",
)

approx_output_list = Dict{Symbol,String}(
    :approximator_recipe => "An `ApproximatorRecipe` object.",
    :approximator_error_recipe => "An `ApproximatorErrorRecipe` object.",
)

approx_method_description = Dict{Symbol,String}(
    :complete_approximator => "A function that generates an `ApproximatorRecipe` given 
    arguments.",
    :update_approximator => "A function that updates the `ApproximatorRecipe` in place
    given the arguments.",
    :rapproximate => "A function that computes a low-rank approximation of the matrix `A`
    using the information in the provided `Approximator` data structure.",
    :complete_approximator_error => "A function that generates an `ApproximatorErrorRecipe`
    given the arguments.",
    :compute_approximator_error => "A function that computes the approximation error of an
    `ApproximatorRecipe` for a matrix `A`.",
)

# Implement the Adjoint structures for the ApproximatorRecipes
"""
    ApproximatorAdjoint{S<:ApproximatorRecipe} <: ApproximatorRecipe 

A structure for the adjoint of an `ApproximatorRecipe`.

### Fields

  - `Parent::ApproximatorRecipe`, the approximator that we compute the adjoint of.
"""
struct ApproximatorAdjoint{S<:ApproximatorRecipe} <: ApproximatorRecipe
    parent::S
end

adjoint(A::ApproximatorRecipe) = ApproximatorAdjoint(A)
# Undo the transpose
adjoint(A::ApproximatorAdjoint{<:ApproximatorRecipe}) = A.parent
# Make transpose wrapper function
transpose(A::ApproximatorRecipe) = ApproximatorAdjoint(A)
# Undo the transpose wrapper
transpose(A::ApproximatorAdjoint{<:ApproximatorRecipe}) = A.parent

# Function skeletons
"""
    complete_approximator(approximator::Approximator, A::AbstractMatrix)

$(approx_method_description[:complete_approximator])

### Arguments
- $(approx_arg_list[:approximator])
- $(approx_arg_list[:A]) 

### Outputs
- $(approx_output_list[:approximator_recipe])
"""
function complete_approximator(approximator::Approximator, A::AbstractMatrix)
    throw(
        ArgumentError(
            "No method `complete_approximator` exists for approximator of type\
            $(typeof(approximator)) and matrix of type $(typeof(A))."
        )
    )
    return nothing
end

"""
    update_approximator!(approximator::ApproximatorRecipe, A::AbstractMatrix)

$(approx_method_description[:update_approximator])

### Arguments
- $(approx_arg_list[:approximator_recipe])
- $(approx_arg_list[:A]) 

### Outputs
- $(approx_output_list[:approximator_recipe])
"""
function update_approximator!(approximator::ApproximatorRecipe, A::AbstractMatrix)
    throw(
        ArgumentError(
            "No method `update_approximator!` exists for approximator of type\
            $(typeof(approximator)) and matrix of type $(typeof(A))."
        )
    )
    return nothing
end

"""
    rapproximate!(approximator::ApproximatorRecipe, A::AbstractMatrix)

$(approx_method_description[:rapproximate])

### Arguments
- $(approx_arg_list[:approximator_recipe])
- $(approx_arg_list[:A]) 

### Outputs
- $(approx_output_list[:approximator_recipe])
"""
function rapproximate!(approximator::ApproximatorRecipe, A::AbstractMatrix)
    throw(
        ArgumentError(
            "No `rapproximate` method exists for approximator of type\
            $(typeof(approximator)) and matrix of type $(typeof(A))."
        )
    )
    return nothing
end

"""
    rapproximate(approximator::Approximator, A::AbstractMatrix)

$(approx_method_description[:rapproximate])

### Arguments
- $(approx_arg_list[:approximator])
- $(approx_arg_list[:A]) 

### Outputs
- $(approx_output_list[:approximator_recipe])
"""
function rapproximate(approximator::Approximator, A::AbstractMatrix)
    approx_recipe = complete_approximator(approximator, A)
    rapproximate!(approx_recipe, A)
    return approx_recipe
end

"""
    complete_approximator_error(
        error::ApproximatorError, 
        approximator::Approximator, 
        A::AbstractMatrix
    )

$(approx_method_description[:complete_approximator_error])

### Arguments
- $(approx_arg_list[:approximator_error])
- $(approx_arg_list[:approximator])
- $(approx_arg_list[:A]) 

### Outputs
- $(approx_output_list[:approximator_error_recipe])
"""
function complete_approximator_error(
    error::ApproximatorError, approximator::Approximator, A::AbstractMatrix
)
    throw(ArgumentError("No `complete_approximator_error! defined for error of type\
    $(typeof(error)), $(typeof(approximator)), and matrix of type $(typeof(A))."))
    return nothing
end

"""
    compute_approximator_error!(
        error::ApproximatorErrorRecipe, 
        approximator::ApproximatorRecipe, 
        A::AbstractMatrix
    )

$(approx_method_description[:compute_approximator_error])

### Arguments
- $(approx_arg_list[:approximator_error_recipe])
- $(approx_arg_list[:approximator_recipe])
- $(approx_arg_list[:A]) 

### Outputs
- Returns the `error::Float64` 
"""
function compute_approximator_error!(
    error::ApproximatorErrorRecipe, approximator::ApproximatorRecipe, A::AbstractMatrix
)
    throw(ArgumentError("No `complete_approximator_error! defined for error of type\
    $(typeof(error)), $(typeof(approximator)), and matrix of type $(typeof(A))."))
    return nothing
end

# Implement a version of the compute error function that works without the recipe
"""
    compute_approximator_error(
        error::ApproximatorError, 
        approximator::ApproximatorRecipe, 
        A::AbstractMatrix
    )

$(approx_method_description[:compute_approximator_error])

### Arguments
- $(approx_arg_list[:approximator_error])
- $(approx_arg_list[:approximator_recipe])
- $(approx_arg_list[:A]) 

### Outputs
- Returns the `error::Float64` 
"""
function compute_approximator_error(
    error::ApproximatorError, approximator::ApproximatorRecipe, A::AbstractMatrix
)
    error_recipe = complete_approximator_error(error, approximator.S, A)
    error_val = compute_approximator_error!(error_recipe, approximator, A)
    return error_val
end
###########################################
# Include the Approximator files
############################################
