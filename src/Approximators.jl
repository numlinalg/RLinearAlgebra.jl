"""
   Approximator 

An abstract supertype for structures that store user defined parameters corresponding to 
    technqiues that form low-rank approximations of the matrix `A`.
"""
abstract type Approximator end

"""
   ApproximatorRecipe

An abstract supertype for structures that store user defined parameters and preallocated 
    memory corresponding to techniques that form low-rank approximations of the matrix `A`.
"""
abstract type ApproximatorRecipe end

"""
    ApproxmatorError

An abstract supertype for structures containing user-defined parameters corresponding to 
methods that evaluate the quality of a Low-Rank approximation of a matrix `A`.
"""
abstract type ApproximatorError end

"""
    ApproxmatorErrorRecipe

An abstract supertype for structures containing user-defined parameters and preallocated 
memory corresponding to methods that evaluate the quality of a Low-Rank approximation of a 
matrix `A`.
"""
abstract type ApproximatorErrorRecipe end

# Implement the Adjoint structures for the ApproximatorRecipes
"""
    ApproximatorAdjoint{S<:ApproximatorRecipe} <: ApproximatorRecipe 

A structure for the adjoint of a approximator recipe.

# Fields
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

A function that uses information in the matrix A and user defined parameters in the 
Approximator to form an ApproximatorRecipe with appropiate memory allocations.

# INPUTS
- `approximator::Approximator`, a data structure containing the user controlled parameters
relating to a particular low rank approximation.
-`A::AbstractMatrix`, a matrix that we wish to approximate.

# OUTPUTS
- `::ApproximatorRecipe`, a data structure with memory preallocated for forming and storing
the desired low rank approximation.
"""
function complete_approximator(approximator::Approximator, A::AbstractMatrix)
    return
end

"""
r_approximate!(approximator::ApproximatorRecipe, A::AbstractMatrix)

A function that computes a Low-Rank approximation of the matrix `A` using the information 
in the provided `ApproximatorRecipe` data structure.

# INPUTS
- `approximator::ApproximatorRecipe`, a data structure for storing the low rank 
approximation to the matrix `A`.
- `A::AbstractMatrix`, the matrix being approximated.

# OUTPUTS
Performs an inplace update of the `ApproximatorRecipe`.
"""
function r_approximate!(approximator::ApproximatorRecipe, A::AbstractMatrix)
    return nothing
end

"""
r_approximate(approximator::Approximator, A::AbstractMatrix)

A function that computes a Low-Rank approximation of the matrix `A` using the information 
in the provided `ApproximatorRecipe` data structure.

# INPUTS
- `approximator::Approximator`, an approximation technique for computing a low-rank 
approximation of the matrix `A`.
- `A::AbstractMatrix`, the matrix being approximated.

# OUTPUTS
- An `ApproximatorRecipe` containing a low rank approximation of the matrix `A`.
"""
function r_approximate(approximator::ApproximatorRecipe, A::AbstractMatrix)
    approx_recipe = complete_approximator(approximator, A)
    r_approximate!(approx_recipe, A)
    return approx_recipe
end

"""
complete_error(error::ApproximatorError, S::CompressorRecipe, A::AbstractMatrix)

A function that produces an `ApproximatorErrorRecipe` from an `ApproximatorError`, 
`CompressorRecipe`, and `AbstractMatrix`.

# INPUTS
- `error::ApproximatorError`, the user controlled parameters associated with the 
approximation error.
- `S::CompressorRecipe`, the compressor information used for the low rank approximation.
- `A::AbstractMatrix`, the matrix being approximated.

# OUTPUTS
- The `ApproximatorErrorRecipe` corresponding to the `ApproximatorError` technique.
"""
function complete_error(
        error::ApproximatorError,
        S::CompressorRecipe,
        A::AbstractMatrix,
    )
    return nothing
end

"""
compute_error(error::ApproximatorErrorMethod, approximator::Approximator, A::AbstractMatrix)

A function that evaluates the quality of an approximation by an approximator.

# INPUTS
- `error::ApproximatorErrorRecipe, ApproximatorErrorRecipe}`, the method for computing the 
approximation error.
- `approx::ApproximatorRecipe`, the low rank of the approximation of the matrix.
- `A::AbstractMatrix`, the matrix.

# OUTPUTS
This function will return an error metric for the approximation of the matrix.
"""
function compute_error(
        error::ApproximatorErrorRecipe,
        approx::ApproximatorRecipe,
        A::AbstractMatrix,
    )
    return nothing
end

# Implement a version of the compute error function that works without the recipe
function compute_error(
        error::ApproximatorError,
        approx::ApproximatorRecipe,
        A::AbstractMatrix,
    )
    error_recipe = complete_error(error, approx.S, A)
    return complete_error(error_recipe, approx, A) 
end
###########################################
# Include the Approximator files
###########################################
