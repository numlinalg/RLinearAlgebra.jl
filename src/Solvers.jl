"""
    Solver

An abstract supertype for structures that contain the user-controlled parameters for 
techniques that solve a linear system ``Ax = b``.
"""
abstract type Solver end

"""
    SolverRecipe

An abstract supertype for structures that contain the user-controlled parameters, linear
system dependent parameters, and preallocated memory for techniques that solve a linear 
system ``Ax = b``.
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

# Docstring Components
solver_arg_list = Dict{Symbol, String}(
    :solver => "`solver::Solver`, a user-specified solver method.",
    :solver_recipe => "`solver::SolverRecipe`, a fully initialized realization for a \
    solver method for a specific linear system.",
    :solver_error => "`solver::SolverError`, a user-specified solver error method.",
    :solver_error_recipe => "`solver::ErrorRecipe`, a fully initialized realization for\
    a solver error method for a specific linear system.",
    :A => "`A::AbstractMatrix`, the coeficient matrix of a linear system.",
    :b => "`b::AbstractVector`, The constant vector for a linear system.",
    :x => " x::AbstractVector`, The solution vector of a linear system."
)

solver_output_list = Dict{Symbol, String}(
    :solver_recipe => "A `SolverRecipe` object.",
    :solver_error_recipe => "A `SolverErrorRecipe` object.",
    :x => "`x::AbstractVector`, the solution to a linear system."
)

solver_method_description = Dict{Symbol, String}(
    :complete_solver => "A function that generates a `SolverRecipe` given the \
    arguments.",
    :complete_solver_error => "A function that generates a `SolverErorRecipe` given the \
    arguments.",
    :compute_solver_error => "A function that evaluates the solution quality of a linear\
    system for a solution vector `x`.",
    :rsolve => "A function that solves a linear system given the arguments."
)
# Function skeletons
"""
    complete_solver(solver::Solver, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)

    $(solver_method_description[:complete_solver])
### Arguments
- $(solver_arg_list[:solver])
- $(solver_arg_list[:x]) 
- $(solver_arg_list[:A]) 
- $(solver_arg_list[:b]) 

### Outputs
- $(solver_output_list[:solver_recipe])
"""
function complete_solver(
        solver::Solver, 
        x::AbstractVector, 
        A::AbstractMatrix,
        b::AbstractVector
    )
    return nothing
end

"""
    rsolve!(
        solver::SolverRecipe, 
        x::AbstractVector, 
        A::AbstractMatrix, 
        b::AbstractVector
    )

    $(solver_method_description[:rsolve])
### Arguments
- $(solver_arg_list[:solver_recipe])
- $(solver_arg_list[:x]) 
- $(solver_arg_list[:A]) 
- $(solver_arg_list[:b]) 

### Outputs
- Returns `Nothing` but updates the `SolverRecipe` and `x` in place.
"""
function rsolve!(
        solver::SolverRecipe,
        x::AbstractVector, 
        A::AbstractMatrix, 
        b::AbstractVector
    ) 
    return nothing
end

"""
    rsolve(
        solver::Solver, 
        x::AbstractVector, 
        A::AbstractMatrix, 
        b::AbstractVector
    )

    $(solver_method_description[:rsolve])
### Arguments
- $(solver_arg_list[:solver_recipe])
- $(solver_arg_list[:x]) 
- $(solver_arg_list[:A]) 
- $(solver_arg_list[:b]) 

### Outputs
- $(solver_output_list[:x])
- $(solver_output_list[:solver_recipe])
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
        solver::Solver,
        A::AbstractMatrix, 
        b::AbstractVector
    )

    $(solver_method_description[:complete_solver_error])
### Arguments
- $(solver_arg_list[:solver_error])
- $(solver_arg_list[:solver])
- $(solver_arg_list[:A]) 
- $(solver_arg_list[:b]) 

### Outputs
- $(solver_output_list[:solver_error_recipe])
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
        solver::SolverRecipe, 
        A::AbstractMatrix, 
        b::AbstractVector
    )

    $(solver_method_description[:compute_solver_error])
### Arguments
- $(solver_arg_list[:solver_error_recipe])
- $(solver_arg_list[:solver_recipe])
- $(solver_arg_list[:A]) 
- $(solver_arg_list[:x]) 
- $(solver_arg_list[:b]) 

### Outputs
- A Float64 representing the progress of the solver. 
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
