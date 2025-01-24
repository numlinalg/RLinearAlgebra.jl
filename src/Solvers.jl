"""
    Solver

An abstract supertype for the user-controlled parameters for 
techniques that solve a linear system `Ax = b`.
"""
abstract type Solver end

"""
    SolverRecipe

An abstract supertype for the user-controlled parameters and memory allocations
for techniques that solve a linear system `Ax = b`.
"""
abstract type SolverRecipe end

"""
    SolverError

An abstract supertype for specifying the user controlled parameters
for a technique that evaluates the quality of solution from a linear solver.
"""
abstract type SolverError end

"""
    SolverErrorRecipe

An abstract supertype for evaluating the quality of solution from a linear solver.
"""
abstract type SolverErrorRecipe end

# Function skeletons
"""
    rsolve!(
        solver::SolverRecipe, 
        x::AbstractVector, 
        A::AbstractMatrix, 
        b::AbstractVector
    )

A function that solvers the linear system `Ax=b` or the least square problem
min_x ||Ax -b||_2^2 using solving method specified by the `solver`
data structure. 

# INPUTS
-`solver::SolverRecipe`, a structure containing all relevant parameter values and memory
to solve a linear system with a speciefied technique.
-`x::AbstractVector`, a vector that will be overwritten with the solution.
-`A::AbstractMatrix`, the coeficent matrix for the linear system.
-`b::AbstractVector`, the constant vector for the linear system.

# OUTPUTS
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

A function that solvers the linear system `Ax=b` or the least square problem
 min_x ||Ax -b||_2^2 using solving method specified by the `solver`
data structure.

# INPUTS
-`solver::Solver`, a solver structure containing all the user defined parameters.
-`x::AbstractVector`, a vector that will be overwritten with the solution.
-`A::AbstractMatrix`, the coeficent matrix for the linear system.
-`b::AbstractVector`, the constant vector for the linear system.

# OUTPUTS
-`x::AbstractVector`, the solution to the linear system.
-`solver_method`, the SolverRecipe generating for applying the desired solving technique.
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
    compute_error(error::ErrorRecipe, solver::Solver, A::AbstractMatrix, b::AbstractVector)

A function that evaluates the quality of a solution from a linear solver.

# INPUTS
- `error::SolverErrorRecipe`, a data structure that stores intermediate information relating 
to the computation of an error metric. For instance, if computing the residual this would be 
a buffer vector for storing the residual.
- `solver::SolverRecipe`, the datastructure containing all the solver information.
- `A::AbstractMatrix`, the coefficient matrix.
- `b::AbstractVector`, the constant vector.

# OUTPUTS
A Float64 representing the progress of the solver. 
"""
function compute_error(
        error::SolverErrorRecipe,
        solver::SolverRecipe,
        A::AbstractMatrix,
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
