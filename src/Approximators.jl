"""
   Approximator 

An abstract supertype for structures that store user-controlled parameters corresponding to 
technqiues that form low-rank approximations of the matrix `A`.
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
    ApproxmatorError

An abstract supertype for structures containing user-controlled parameters corresponding to 
methods that evaluate the quality of a Low-Rank approximation of a matrix `A`.
"""
abstract type ApproximatorError end

"""
    ApproxmatorErrorRecipe

An abstract supertype for structures containing user-controlled parameters, linear system
dependent parameters and preallocated memory corresponding to methods that evaluate the 
quality of a Low-Rank approximation of a matrix `A`.
"""
abstract type ApproximatorErrorRecipe end

# Implement the Adjoint structures for the ApproximatorRecipes
"""
    ApproximatorAdjoint{S<:ApproximatorRecipe} <: ApproximatorRecipe 

A structure for the adjoint of a approximator recipe.

### Fields
-`Parent::ApproximatorRecipe`, the approximator that we compute the adjoint of.
"""
struct ApproximatorAdjoint{S<:ApproximatorRecipe} <: ApproximatorRecipe
    parent::S
end

Adjoint(A::ApproximatorRecipe) = ApproximatorAdjoint{typeof(A)}(A)
adjoint(A::ApproximatorRecipe) = Adjoint(A)
# Undo the transpose
adjoint(A::ApproximatorAdjoint{<:ApproximatorRecipe}) = A.parent
# Make transpose wrapper function
transpose(A::ApproximatorRecipe) = Adjoint(A)
# Undo the transpose wrapper
transpose(A::ApproximatorAdjoint{<:ApproximatorRecipe}) = A.parent

# Function skeletons
"""
    complete_approximator(approximator::Approximator, A::AbstractMatrix)

A function that uses information in the matrix `A` and user-controlled parameters in the 
Approximator to form an `ApproximatorRecipe` with appropiate memory allocations.

### Arguments
- `approximator::Approximator`, a data structure containing the user-controlled parameters
relating to a particular low rank approximation.
-`A::AbstractMatrix`, a matrix that we wish to approximate.

### Outputs
returns an `ApproximatorRecipe` with memory preallocated for forming and storing
the desired low rank approximation.
"""
function complete_approximator(approximator::Approximator, A::AbstractMatrix)
    return
end

"""
    rapproximate!(approximator::ApproximatorRecipe, A::AbstractMatrix)

A function that computes a Low-Rank approximation of the matrix `A` using the information 
in the provided `ApproximatorRecipe` data structure.

### Arguments
- `approximator::ApproximatorRecipe`, a data structure for storing the low rank 
approximation to the matrix `A`.
- `A::AbstractMatrix`, the matrix being approximated.

### Outputs
Performs an inplace update of the `ApproximatorRecipe`.
"""
function rapproximate!(approximator::ApproximatorRecipe, A::AbstractMatrix)
    return nothing
end

"""
    rapproximate(approximator::Approximator, A::AbstractMatrix)

A function that computes a Low-Rank approximation of the matrix `A` using the information 
in the provided `Approximator` data structure.

### Arguments
- `approximator::Approximator`, an approximation technique for computing a low-rank 
approximation of the matrix `A`.
- `A::AbstractMatrix`, the matrix being approximated.

### Outputs
- An `ApproximatorRecipe` containing a low rank approximation of the matrix `A`.
"""
function rapproximate(approximator::ApproximatorRecipe, A::AbstractMatrix)
    approx_recipe = complete_approximator(approximator, A)
    r_approximate!(approx_recipe, A)
    return approx_recipe
end

"""
   complete_approximator_error(
        error::ApproximatorError, 
        S::CompressorRecipe, 
        A::AbstractMatrix
    )

A function that produces an `ApproximatorErrorRecipe` from an `ApproximatorError`, 
`CompressorRecipe`, and `AbstractMatrix`.

### Arguments
- `error::ApproximatorError`, the user controlled parameters associated with the 
approximation error.
- `S::CompressorRecipe`, the compressor information used for the low rank approximation.
- `A::AbstractMatrix`, the matrix being approximated.

### Outputs
- The `ApproximatorErrorRecipe` corresponding to the `ApproximatorError` technique.
"""
function complete_approximator_error(
        error::ApproximatorError,
        S::CompressorRecipe,
        A::AbstractMatrix,
    )
    return nothing
end

"""
compute_approximator_error(
    error::ApproximatorErrorMethod, 
    approximator::Approximator, 
    A::AbstractMatrix
)

A function that evaluates the quality of an `ApproximatorRecipe`.

### Arguments
- `error::ApproximatorErrorRecipe`, the method for computing the 
approximation error.
- `approx::ApproximatorRecipe`, the low rank of the approximation of the matrix.
- `A::AbstractMatrix`, the matrix.

### Outputs
This function will return an error metric for the approximation of the matrix.
"""
function compute_approximator_error(
        error::ApproximatorErrorRecipe,
        approx::ApproximatorRecipe,
        A::AbstractMatrix,
    )
    return nothing
end

# Implement a version of the compute error function that works without the recipe
function compute_approximator_error(
        error::ApproximatorError,
        approx::ApproximatorRecipe,
        A::AbstractMatrix,
    )
    error_recipe = complete_approximator_error(error, approx.S, A)
    return complete_approximator_error(error_recipe, approx, A) 
end
###########################################
# Include the Approximator files
###########################################
