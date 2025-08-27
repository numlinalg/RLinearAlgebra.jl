"""
    LSgradient <: SolverError

A `SolverError` structure for computing the least-squares gradient,
    ``\\nabla f(x) = A' (A x - b)``

# Fields
- None
"""
struct LSgradient <: SolverError end

"""
    LSgradientRecipe <: SolverErrorRecipe
A `SolverErrorRecipe` structure for computing the gradient of least-squares objective.

# Fields
- `gradient::AbstractVector`, `A'r`.
"""
mutable struct LSgradientRecipe{V<:AbstractVector} <: SolverErrorRecipe
    gradient::V
end

function complete_error(
    error::LSgradient, 
    solver::Solver, 
    A::AbstractMatrix, 
    b::AbstractVector
)
    gradient = zeros(size(A,2))
    return LSgradientRecipe{typeof(b)}(gradient)
end

function compute_error(
    error::LSgradientRecipe, 
    solver::SolverRecipe, 
    A::AbstractMatrix, 
    b::AbstractVector
)::Float64
    mul!(error.gradient, A', solver.residual_vec, 1.0, 0.0) # grad = A'r
    return norm(error.gradient)
end
