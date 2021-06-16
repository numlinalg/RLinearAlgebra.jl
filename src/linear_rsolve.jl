############################################################################################
## This file is part of RLinearAlgebra.jl
##
## Overview: abstractions and methods for performing various deploying Random Linear System
## solvers
##
## Contents
## - `RLSSolver` struct
## - `rsolve` functions
## - Randomized linear Solvers
## - Export Statements
##
############################################################################################

# Dependencies:

#############################################
# `RLSSolver` stucture
#############################################

"""
    RLSSolver{S<:LinSysSampler, R<:LinSysSolveRoutine,
        L<:LinSysSolverLog, C<:LinSysStopCriterion}

An encapsulation for a (R)andomized (L)inear (S)ysterm Solver as a mutable structure.

# Fields
- `sampler::S`, a procedure for sampling, sketching or selecting with respect to a linear
    system.
- `routine::R`, a routine for randomly generating the subproblem generated by the sampling,
    sketching or selecting operation
- `log::L`, a logger for the progress and behavior of the randomized linear system solver.
- `stop::C`, a stopping criterion for the randomized linear system solver.
- `x::Union{Vector{Float64},Nothing}`, the result produced by the randomized linear system
    solver.

# Constructions
- `RLSSolver(iter::Int64)` specifies a solver with: random cyclic row sampling;  partially
    orthogonalized row projections (memory of five) subproblem solver; a full residual
    logger; and a maximum iteration stopping criterion specified by `iter`.
"""
mutable struct RLSSolver{S<:LinSysSampler, R<:LinSysSolveRoutine,
    L<:LinSysSolverLog, C<:LinSysStopCriterion}

    sampler::S
    routine::R
    log::L
    stop::C
    x::Union{Vector{Float64},Nothing}
end

RLSSolver(iter::Int64) = RLSSolver(
    LinSysVecRowRandCyclic(),   # Random Cyclic Sampling
    LinSysVecRowProjPO(),       # Partially Orthogonalized Row Projection, 5 vector memory
    LSLogFull(),                # Full Logger: maintains residual history
    LSStopMaxIterations(iter),  # Maximum iterations stopping criterion
    nothing                     # System solution
)


#############################################
# `rsolve` functions
#############################################
"""
    rsolve([solver::RLSSolver,] A, b)

Deploys an iterative randomized solver for the linear system whose coefficient matrix is
    encapsulated in `A` and whose constant vector is encapsulated in `b`. If the `solver`
    is not specified, then the default `RLSSolver` is constructed and used with a maximum
    number of iterations of `10 * length(b)`. See `RLSSolver` for more details on the
    default solver.

The function returns the solution.
"""
function rsolve(A, b)
    solver = RLSSolver(10 * length(b))
    x = zeros(size(A, 2))
    rsolve!(solver, A, b, x)

    return x
end

function rsolve(solver::RLSSolver, A, b)
    x = zeros(size(A, 2))
    rsolve!(solver, A, b, x)

    return x
end

"""
    rsolve!([solver::RLSSolver,] A, b, x::AbstractVector)

Identical to `rsolve` except an initial iterate is supplied by the argument `x`. The
    function overwrites the solution into the vector `x`.
"""
function rsolve!(A, b, x::AbstractVector)
    solver = RLSSolver(10 * length(b))
    rsolve!(solver, A, b, x)

    return nothing
end

function rsolve!(solver::RLSSolver, A, b, x::AbstractVector)

    iter = 0
    log_update!(solver.log, solver.sampler, x, (), iter, A, b)

    while check_stop_criterion(solver.log, solver.stop) == false
        iter += 1
        samp = sample(solver.sampler, A, b, x, iter)
        rsubsolve!(solver.routine, x, samp, iter)
        log_update!(solver.log, solver.sampler, x, samp, iter, A, b)
    end

    solver.x = x

    return nothing
end

#############################################
# Randomized Linear Solvers
#############################################


#############################################
# Export Statements
#############################################
# See RLinearAlgebra.jl
#export RLSSolver, rsolve, rsolve!
