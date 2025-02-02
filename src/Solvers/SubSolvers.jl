"""
    SubSolver

An abstract supertype for structures that contain user-controlled parameters 
for a linear solver applied to compressed matrix blocks. 
"""
abstract type SubSolver end

"""
    SubSolverRecipe

An abstract supertype for structures that contain user-controlled parameters, linear system
specific parameters and preallocated memory for a linear solver applied to compressed matrix
blocks. 
"""
abstract type SubSolverRecipe end

"""
    complete_sub_solver(solver::SubSolver, A::AbstractMatrix)

A function that takes the user-controlled parameters from the `SubSolver` data structure 
and the matrix A and uses this information to create the `SubSolverRecipe`.

### Arguments
- `solver::SubSolver`, the `SubSolver` structure that contains the user controlled 
parameters.
- `A::AbstractMatrix`, the matrix that the `SubSolverRecipe` will contain.

### Outputs
- Will return a SubSolverRecipe that can be applied to a vector.
"""
function complete_sub_solver(solver::SubSolver, A::AbstractMatrix)
    return 
end


"""
    update_sub_solver!(solver::SubSolverRecipe, A::AbstractMatrix)

A function that updates the structure of `SubSolverRecipe` with information from the matrix
A. These updates typically require preforming decompositons or updating pointers.

### Arguments
- `solver::SubSolverRecipe`, the `SubSolverRecipe` structure that can be applied 
to a matrix or vector.
- `A::AbstractMatrix`, the matrix that the `SubSolverRecipe` will contain.

### Outputs
- Modifies the `SubSolverRecipe` in place and returns nothing.
"""
function update_sub_solver!(solver::SubSolverRecipe, A::AbstractMatrix)
    return
end

function ldiv!(x::AbstractVector, solver::SubSolverRecipe, b::AbstractVector)
    return
end

###########################################
# Include SubSolver files
###########################################
