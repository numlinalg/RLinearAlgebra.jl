"""
    LQSolver <: UnderdeterminedSolver

#Fields
-`A::Union{Nothing, AbstractMatrix}`, A matrix to be solved. 

Function for lq inner solver. 
"""
mutable struct LQSolver <: UnderdeterminedSolver
    A::Union{Nothing, AbstractMatrix}
end

LQSolver() = LQSolver(nothing)

function initializeSolver!(S::LQSolver, A::AbstractMatrix)
    S.A = A
end

function updateSolver!(S::LQSolver, A::AbstractMatrix)
    S.A = A
end

function resetSolver!(S::LQSolver)
    return nothing
end

function ldiv!(y::AbstractVector, S::LQSolver, x::AbstractVector)
    # Fill y with zeros. 
    fill!(y, 0)
    ldiv!(y, lq(S.A), x)
end
