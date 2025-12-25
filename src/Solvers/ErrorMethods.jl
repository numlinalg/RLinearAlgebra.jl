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
    :complete_error => "A function that generates a `SolverErrorRecipe` given the 
    arguments.",
    :compute_error => "A function that evaluates the error for a proposed solution 
    vector.",
)
# Function skeletons

"""
    complete_error(
        error::SolverError, 
        solver::Solver,
        A::AbstractMatrix, 
        b::AbstractVector
    )

$(solver_method_description[:complete_error])

# Arguments
- $(solver_arg_list[:solver_error])
- $(solver_arg_list[:solver])
- $(solver_arg_list[:A]) 
- $(solver_arg_list[:b]) 

# Returns 
- $(solver_output_list[:solver_error_recipe])
"""
function complete_error(
    error::SolverError, 
    solver::Solver, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    return throw(
        ArgumentError(
            "There is no `complete_error` method defined for a $(typeof(error)) \
            `SolverError`,  $(typeof(solver)) `SolverRecipe`, $(typeof(A)), and \
            $(typeof(b))."
        )
    )
end

"""
    compute_error(
        error::SolverErrorRecipe, 
        solver::SolverRecipe, 
        x::AbstractVector,
        A::AbstractMatrix, 
        b::AbstractVector
    )

$(solver_method_description[:compute_error])

# Arguments
- $(solver_arg_list[:solver_error_recipe])
- $(solver_arg_list[:solver_recipe])
- $(solver_arg_list[:x]) 
- $(solver_arg_list[:A]) 
- $(solver_arg_list[:b]) 

# Returns 
-  Returns `nothing`
"""
function compute_error(
    error::SolverErrorRecipe,
    solver::SolverRecipe,
    x::AbstractVector,
    A::AbstractMatrix,
    b::AbstractVector,
)
    return throw(
        ArgumentError(
            "No `compute_error` method defined for a $(typeof(error)) `SolverErrorRecipe`,\
            $(typeof(solver)) `SolverRecipe`, $(typeof(x)), $(typeof(A)), and $(typeof(b))."
        )
    )
end

# Include error method files 
include("ErrorMethods/full_residual.jl")
include("ErrorMethods/ls_gradient.jl")
