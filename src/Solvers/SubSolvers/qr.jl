"""
    QRSolver <: SubSolver

A type containing information relevant to solving the linear subsystems created by the 
    Solver routines with the LQ factorization. As there are no user controlled parameters, 
    if the user wishes to use this method they can simply specify `QRSolver()`.
"""
struct QRSolver <: SubSolver

end

"""
    QRSolverRecipe <: SubSolverRecipe{M<:AbstractArray}

A mutable type containing informtation relevant to solving the linear subsytems created by 
    the Solver routines with the QR factorization.

# Fields
 - `A::M`, The matrix in the linear system that will be solved with the QR solver.
"""
mutable struct QRSolverRecipe{M<:AbstractArray} <: SubSolverRecipe
    A::M
end

function complete_sub_solver(solver::QRSolver, A::AbstractMatrix)
    return QRSolverRecipe{typeof(A)}(A) 
end

function complete_sub_solver(solver::QRSolver, A::AbstractVector)
    return QRSolverRecipe{typeof(A)}(A) 
end

function update_sub_solver!(solver::QRSolverRecipe, A::AbstractMatrix)
    # This will overwrite the block of the matrix so do not reuse the block values
    solver.A = A
    return nothing
end

function ldiv!(
    x::AbstractVector, 
    solver::QRSolverRecipe{<:AbstractMatrix}, 
    b::AbstractVector
)
    fill!(x, zero(eltype(b)))
    # this will modify B in place so you cannot use it again
    ldiv!(x, qr!(solver.A), b)
    return nothing
end