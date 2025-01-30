"""
    SubSolver

An abstract supertype for user-controlled parameters for a block solver. 
"""
abstract type SubSolver end

"""
    SubSolverRecipe

An abstract supertype for all information related to a specific solver. This includes both 
the user controlled parameters defined in the `SubSolver` and memory structures specific
to the solver.
"""
abstract type SubSolverRecipe end

"""
    complete_sub_solver(solver::SubSolver, A::AbstractMatrix)

A function that takes the user defined parameters from the `SubSolver` data structure 
and the matrix A and uses this information to create the `SubSolverRecipe`.

# INPUTS
- `solver::SubSolver`, the `SubSolver` structure that contains the user controlled 
parameters.
- `A::AbstractMatrix`, the matrix that the `SubSolverRecipe` will contain.

# OUTPUTS
Will return a SubSolverRecipe that can be applied to a vector.
"""
function complete_sub_solver(solver::SubSolver, A::AbstractMatrix)
    return <:SubSolverRecipe
end


"""
    update_sub_solver!(solver::SubSolverRecipe, A::AbstractMatrix)

A function that updates the structure of subSolver with the matrix A. These updates
typically require preforming decompositons or updating pointers.

# INPUTS
- `solver::SubSolverRecipe`, the `SubSolverRecipe` structure that can be applied 
to a matrix or vector..
- `A::AbstractMatrix`, the matrix that the `SubSolverRecipe` will contain.
"""
function update_sub_solver!(solver::SubSolverRecipe, A::AbstractMatrix)
    return
end

"""
    function ldiv!(x::AbstractVector, solver::LQSolverRecipe, b::AbstractVector)

A function that uses a subSolver method to find the `x` corresponding to the constant
vector `b`.
"""
function LinearAlgebra.ldiv!(x::AbstractVector, solver::SubSolverRecipe, b::AbstractVector)
    return
end

###########################################
# Include SubSolver files
###########################################
