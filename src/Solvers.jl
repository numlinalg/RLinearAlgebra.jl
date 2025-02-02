"""
    Solver

An abstract supertype for structures that contain the user-controlled parameters for 
techniques that solve a linear system `Ax = b`.
"""
abstract type Solver end

"""
    SolverRecipe

An abstract supertype for structures that contain the user-controlled parameters, linear
system dependent parameters, and preallocated memory for techniques that solve a linear 
system `Ax = b`.
"""
abstract type SolverRecipe end

"""
    SolverError

An abstract supertype for structures that contain the user-controlled parameters
for techniques that evaluates the quality of solution from a linear solver.
"""
abstract type SolverError end

"""
    SolverErrorRecipe

An abstract supertype for structures that contain the user-controlled parameters, linear 
system dependent parameters, and preallocated memory for techniques that evaluate the 
solution to a liear solver..
"""
abstract type SolverErrorRecipe end

# Function skeletons
"""
    complete_solver(solver::Solver, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)

A function that combines the information in the `Solver` data structure, matrix `A`, vector
`x`, and vector `b` to for a `SolverRecipe` which can be used to solve the linear system.

### Arguments
- `solver::Solver`, a solver structure containing all the user defined parameters.
- `x::AbstractVector`, a vector that will be overwritten with the solution.
- `A::AbstractMatrix`, the coeficent matrix for the linear system.
- `b::AbstractVector`, the constant vector for the linear system.

### Outputs
Returns a `SolverRecipe`.
"""
function complete_solver(
        solver::Solver, 
        x::AbstractVector, 
        A::AbstractMatrix,
        b::AbstractVector
    )
    return 
end

"""
    rsolve!(
        solver::SolverRecipe, 
        x::AbstractVector, 
        A::AbstractMatrix, 
        b::AbstractVector
    )

A function that solve the linear system `Ax=b` or the least square problem
min_x ||Ax -b||_2^2 using solving method specified by the `solver`
data structure. 

### Arguments
-`solver::SolverRecipe`, a structure containing all relevant parameter values and memory
to solve a linear system with a speciefied technique.
- `x::AbstractVector`, a vector that will be overwritten with the solution.
- `A::AbstractMatrix`, the coeficent matrix for the linear system.
- `b::AbstractVector`, the constant vector for the linear system.

### Outputs
The function updates `solver` and `x` in-place.
"""
function rsolve!(
        solver::SolverRecipe,
        x::AbstractVector, 
        A::AbstractMatrix, 
        b::AbstractVector
    ) 
    return
end

"""
    rsolve(
        solver::Solver, 
        x::AbstractVector, 
        A::AbstractMatrix, 
        b::AbstractVector
    )

A function that solves the linear system `Ax=b` or the least square problem
 min_x ||Ax -b||_2^2 using solving method specified by the `solver`
data structure.

### Arguments
- `solver::Solver`, a solver structure containing all the user defined parameters.
- `x::AbstractVector`, a vector that will be overwritten with the solution.
- `A::AbstractMatrix`, the coeficent matrix for the linear system.
- `b::AbstractVector`, the constant vector for the linear system.

### Outputs
- `x::AbstractVector`, the solution to the linear system.
- `solver_method`, the SolverRecipe generating for applying the desired solving technique.
"""
function rsolve(
        solver::Solver, 
        x::AbstractVector, 
        A::AbstractMatrix, 
        b::AbstractVector
    ) 
    solver_method = complete_solver(solver, x, A, b) 
    rsolve!(solver_method, x, A, b)
    return x, solver_method
end

"""
    complete_solver_error(
        error::SolverError, 
        ::AbstractMatrix, 
        ::AbstractVector
    )

A function that generates a SolverErrorRecipe using the user defined inputs of a SolverError
and information from the matrix `A` and vector `b`.

### Arguments
- `error::SolverError`, a data structure that stores the user defined parameters relating to 
an error method.
- `A::AbstractMatrix`, the coefficient matrix.
- `b::AbstractVector`, the constant vector.

### Outputs
A Float64 representing the progress of the solver. 
"""
function complete_solver_error(
        error::SolverErrorRecipe,
        solver::SolverRecipe,
        A::AbstractMatrix,
        b::AbstractVector
    )
    return nothing
end

"""
    compute_solver_error(
        error::ErrorRecipe, 
        solver::Solver, 
        A::AbstractMatrix, 
        b::AbstractVector
    )

A function that evaluates the quality of a solution from a linear solver.

### Arguments
- `error::SolverErrorRecipe`, a data structure that stores intermediate information relating 
to the computation of an error metric. For instance, if computing the residual this would be 
a buffer vector for storing the residual.
- `solver::SolverRecipe`, the datastructure containing all the solver information.
- `A::AbstractMatrix`, the coefficient matrix.
- `b::AbstractVector`, the constant vector.

### Outputs
A Float64 representing the progress of the solver. 
"""
function compute_solver_error(
        error::SolverErrorRecipe,
        solver::SolverRecipe,
        A::AbstractMatrix,
        x::AbstractVector,
        b::AbstractVector
    )
    return nothing
end

############################
# The Loggers
############################
include("Solvers/Loggers.jl")
#############################
# The sub solvers
#############################
include("Solvers/SubSolvers.jl")
#############################
# The Solver Routine Files
############################
