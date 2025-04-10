"""
    LQSolver <: SubSolver

A type containing informtation relevant to solving the linear subsystems created by the 
    Solver routines with the LQ factorization. As there are no user controlled parameters, 
    if the user wishes to use this method they can simply specify `LQSolver()`.
"""
struct LQSolver <: SubSolver

end

"""
    LQSolverRecipe <: SubSolverRecipe{M<:AbstractArray}

A mutable type containing informtation relevant to solving the linear subsytems created by 
    the Solver routines with the LQ factorization.

# Fields
- A::M, The matrix in the linear system that will be solved with 
the LQ solver.
"""
mutable struct LQSolverRecipe{M<:AbstractArray} <: SubSolverRecipe
    A::M
end

function complete_sub_solver(solver::LQSolver, A::AbstractMatrix)
    return LQSolverRecipe{typeof(A)}(A) 
end

function complete_sub_solver(solver::LQSolver, A::AbstractVector)
    return LQSolverRecipe{typeof(A)}(A) 
end

function update_sub_solver!(solver::LQSolverRecipe, A::AbstractArray)
    # This will overwrite the block of the matrix so do not reuse the block values
    solver.A = A
    return nothing
end

function ldiv!(
    x::AbstractVector, 
    solver::LQSolverRecipe{<:AbstractMatrix}, 
    b::AbstractVector
)
    fill!(x, zero(eltype(b)))
    # this will modify B in place so you cannot use it again
    ldiv!(x, lq!(solver.A), b)
    return nothing
end

function ldiv!(
    x::AbstractVector, 
    solver::LQSolverRecipe{<:AbstractVector}, 
    b::Real
)
    # compute the kaczmarz update vector when given a 1-d array(should copy to x beforehand)
    x .= (b - dot(solver.A, x)) / dot(solver.A, solver.A) * solver.A
    return nothing
end
