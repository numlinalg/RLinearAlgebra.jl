"""
    Solver

An abstract supertype for structures that contain the user-controlled parameters for
methods that solve linear systems and least squares problems.
"""
abstract type Solver end

"""
    SolverRecipe

An abstract supertype specifying a solver method with pre-allocated data structures given a 
coefficient matrix and constant vector.
"""
abstract type SolverRecipe end

"""
    SolverError

An abstract supertype for structures that track and/or evaluate the quality of a solution
for a linear system or least squares.
"""
abstract type SolverError end

"""
    SolverErrorRecipe

An abstract supertype for structures that contain the user-controlled parameters, linear
system dependent parameters, and preallocated memory for techniques that evaluate the
solution to a linear solver.
"""
abstract type SolverErrorRecipe end

# Docstring Components
solver_arg_list = Dict{Symbol,String}(
    :solver => "`solver::Solver`, a user-specified solver method.",
    :solver_recipe => "`solver::SolverRecipe`, a fully initialized realization for a 
    solver method for a specific linear system.",
    :solver_error => "`error::SolverError`, a user-specified solver error method.",
    :solver_error_recipe => "`error::SolverErrorRecipe`, a fully initialized realization for
    a solver error method for a specific linear system or least squares problem.",
    :A => "`A::AbstractMatrix`, a coefficient matrix.",
    :b => "`b::AbstractVector`, a constant vector.",
    :x => "`x::AbstractVector`, a vector for the proposed solution.",
)

solver_output_list = Dict{Symbol,String}(
    :solver_recipe => "A `SolverRecipe` object.",
    :solver_error_recipe => "A `SolverErrorRecipe` object.",
    :x => "`x::AbstractVector`, a proposed solution to a linear system or least squares 
    problem.",
)

solver_method_description = Dict{Symbol,String}(
    :complete_solver => "A function that generates a `SolverRecipe` given the 
    arguments.",
    :complete_solver_error => "A function that generates a `SolverErrorRecipe` given the 
    arguments.",
    :compute_solver_error => "A function that evaluates the error for a proposed solution 
    vector.",
    :rsolve => "A function that solves a linear system given the arguments.",
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
    solver::Solver, x::AbstractVector, A::AbstractMatrix, b::AbstractVector
)
    throw(
        ArgumentError("There is no `complete_solver` method specified for a 
                      $(typeof(solver)) solver,$(typeof(x)), $(typeof(A)), 
                      and $(typeof(b))."
        )
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
- Returns `nothing`. Updates the `SolverRecipe` and `x` in place.
"""
function rsolve!(
    solver::SolverRecipe, x::AbstractVector, A::AbstractMatrix, b::AbstractVector
)
    throw(
        ArgumentError("There is no `rsolve!` method specified for a $(typeof(solver)) 
                      solver, $(typeof(x)), $(typeof(A)), and $(typeof(b))."
        )
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
- $(solver_arg_list[:solver])
- $(solver_arg_list[:x]) 
- $(solver_arg_list[:A]) 
- $(solver_arg_list[:b]) 

### Outputs
- $(solver_output_list[:x])
- $(solver_output_list[:solver_recipe])
"""
function rsolve(solver::Solver, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    solver_method = complete_solver(solver, x, A, b)
    rsolve!(solver_method, x, A, b)
    return x, solver_method
end

"""
    complete_solver_error(
        error::SolverError, 
        solver::SolverRecipe,
        A::AbstractMatrix, 
        b::AbstractVector
    )

$(solver_method_description[:complete_solver_error])

### Arguments
- $(solver_arg_list[:solver_error])
- $(solver_arg_list[:solver_recipe])
- $(solver_arg_list[:A]) 
- $(solver_arg_list[:b]) 

### Outputs
- $(solver_output_list[:solver_error_recipe])
"""
function complete_solver_error(
    error::SolverError, solver::SolverRecipe, A::AbstractMatrix, b::AbstractVector
)
    throw(
        ArgumentError("There is no `complete_solver_error` method specified for 
                      a $(typeof(error)) `SolverError`,  $(typeof(solver)) `SolverRecipe`, 
                      $(typeof(A)), and $(typeof(b))."
        )
    )
    return nothing
end

"""
    compute_solver_error(
        error::SolverErrorRecipe, 
        solver::SolverRecipe, 
        A::AbstractMatrix, 
        b::AbstractVector
    )

$(solver_method_description[:compute_solver_error])

### Arguments
- $(solver_arg_list[:solver_error_recipe])
- $(solver_arg_list[:solver_recipe])
- $(solver_arg_list[:x]) 
- $(solver_arg_list[:A]) 
- $(solver_arg_list[:b]) 

### Outputs
-  Returns `nothing`
"""
function compute_solver_error(
    error::SolverErrorRecipe,
    solver::SolverRecipe,
    x::AbstractVector,
    A::AbstractMatrix,
    b::AbstractVector,
)
    throw(
        ArgumentError("There is no `complete_solver_error` method specified for 
                      a $(typeof(error)) `SolverErrorRecipe`,  
                      $(typeof(solver)) `SolverRecipe`, 
                      $(typeof(x)), $(typeof(A)), and $(typeof(b))."
        )
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
