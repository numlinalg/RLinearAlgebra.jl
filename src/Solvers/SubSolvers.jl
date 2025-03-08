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

# Docstring Components
sub_solver_arg_list = Dict{Symbol,String}(
    :sub_solver => "`solver::SubSolver`, a user-specified sub-solving method.",
    :sub_solver_recipe => "`solver::SubSolverRecipe`, a fully initialized realization for a
    logging method for a specific linear sub solver.",
    :A => "`A::AbstractMatrix`, a target matrix for compression.",
)

sub_solver_output_list = Dict{Symbol,String}(
    :sub_solver_recipe => "A `SubSolverRecipe` object."
)

sub_solver_method_description = Dict{Symbol,String}(
    :complete_sub_solver => "A function that generates a `SubSolverRecipe` given the 
    arguments.",
    :update_sub_solver => "A function that updates the `SubSolver` in place given 
    arguments.",
)
"""
    complete_sub_solver(solver::SubSolver, A::AbstractMatrix)

$(sub_solver_method_description[:complete_sub_solver])

### Arguments
- $(sub_solver_arg_list[:sub_solver])
- $(sub_solver_arg_list[:A]) 

### Outputs
- $(sub_solver_output_list[:sub_solver_recipe])
"""
function complete_sub_solver(solver::SubSolver, A::AbstractMatrix)
    return nothing
end

"""
    update_sub_solver!(solver::SubSolverRecipe, A::AbstractMatrix)

$(sub_solver_method_description[:update_sub_solver])

### Arguments
- $(sub_solver_arg_list[:sub_solver])
- $(sub_solver_arg_list[:A]) 

### Outputs
- Modifies the `SubSolverRecipe` in place and returns nothing.
"""
function update_sub_solver!(solver::SubSolverRecipe, A::AbstractMatrix)
    return nothing
end

function ldiv!(x::AbstractVector, solver::SubSolverRecipe, b::AbstractVector)
    return nothing
end

###########################################
# Include SubSolver files
###########################################
