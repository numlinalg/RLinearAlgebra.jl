############################################################################################
## This file is part of RLinearAlgebra.jl
##
## Overview: abstractions and methods for routines used in solving linear systems of
## equations.
##
## Contents
## - Abstract Types
## - `rsubsolve!` function documentation
## - Vector Row Projection Solvers
## - Vector Column Projection Solvers
## - Block Row Projection Solvers
## - Block Column Projection Solvers
## - Preconditioned Krylov Solvers
## - Export Statements
##
############################################################################################

# Dependencies: LinearAlgebra

#############################################
# Abstract Types
#############################################
"""
    LinSysSolveRoutine

Abstract supertype that specifies the type of linear system solver routine being deployed.
"""
abstract type LinSysSolveRoutine end

"""
    LinSysVecRowProjection <: LinSysSolveRoutine

Abstract supertype for vector row action projection methods.
"""
abstract type LinSysVecRowProjection <: LinSysSolveRoutine end

"""
    LinSysVecColProjection <: LinSysSolveRoutine

Abstract supertype for vector column action projection methods.
"""
abstract type LinSysVecColProjection <: LinSysSolveRoutine end

"""
    LinSysBlkRowProjection <: LinSysSolveRoutine

Abstract supertype for block row action projection methods.
"""
abstract type LinSysBlkRowProjection <: LinSysSolveRoutine end

"""
    LinSysBlkColProjection <: LinSysSolveRoutine

Abstract supertype for block column action projection methods.
"""
abstract type LinSysBlkColProjection <: LinSysSolveRoutine end

"""
    LinSysPreconKrylov <: LinSysSolveRoutine

Abstract supertype for preconditioned Krylov solver.
"""
abstract type LinSysPreconKrylov <: LinSysSolveRoutine end

#############################################
# `rsubsolve!` Function Documentation`
#############################################
"""
    rsubsolve!(
        type::LinSysSolveRoutine,
        x::AbstractVector,
        samp::Tuple,
        iter::Int64)

A common interface for specifying different strategies for solving a subproblem generated
    by a sampling, selecting or sketching operation on a linear system. The `type` argument
    is used to select the appropriately defined strategy. The argument `x` is the current
    iterate value for the solution. The argument `samp` depends on the subtype
    of `LinSysSolveRoutine` that is being deployed and is described below. The `iter`
    argument is the iteration counter.

The value of `samp` depends on the type of subtype of `LinSysSolveRoutine` being deployed.
To describe this, let `A` be the coefficient matrix of the system, and `b` its constant
vector.
- For `T<:LinSysVecRowProjection`, `samp` is a two-dimensional tuple where the first
    entry is a vector in the row space of `A`; and the second entry is a scalar value
    corresponding to a linear combination of the elements of `b`.
- For `T<:LinSysVecColProjection`, `samp` is a three-dimensional tuple where the first
    entry is a vector of `length(x)` corresponding to the search direction; the second
    entry is a matrix with the same number of rows as `A` (usually it is `A`);
    and the third entry is a scalar-valued residual for the normal system corresponding
    to `samp[1]'* A' * (A * x - b)`.

The function `rsubsolve!` updates the quantity `x` and any fields of `type` that must be
    updated.
"""
function rsubsolve!(
    type::Nothing,
    x::AbstractVector,
    samp::Tuple,
    iter::Int64,
)
    return nothing
end
############################################
# Linear Solvers Helpers
############################################
include("linear_solver_routines/block_solver_helpers/gentlemans_house.jl")
include("linear_solver_routines/arnoldi_solver_helpers/solve_hessenberg_system.jl")
#############################################
# Vector Row Projection Solvers
#############################################
include("linear_solver_routines/vec_row_projection_std.jl")
include("linear_solver_routines/vec_row_projection_portho.jl")
include("linear_solver_routines/vec_row_projection_fortho.jl")

#############################################
# Vector Column Projection Solvers
#############################################
include("linear_solver_routines/vec_col_projection_std.jl")
include("linear_solver_routines/vec_col_projection_portho.jl")
include("linear_solver_routines/vec_col_projection_fortho.jl")

#############################################
# Block Row Projection Solvers
#############################################
# Exact Solvers
include("linear_solver_routines/block_row_projection.jl")
# Approximate Solvers
include("linear_solver_routines/iterative_hessian_sketch.jl")

#############################################
# Block Column Projection Solvers
#############################################
# Exact Solvers
include("linear_solver_routines/block_col_projection.jl")
# Approximate Solvers

#############################################
# Precondition Krylov Solvers
#############################################
#Blendenpik

#############################################
# Alternative Solvers
#############################################
include("linear_solver_routines/randomized_arnoldi_solver.jl")
include("linear_solver_routines/arnoldi_solver.jl")

#############################################
# Export Statements
#############################################
# See RLinearAlgebra.jl
#export LinSysSolveRoutine, LinSysVecRowProjection, LinSysVecColProjection,
#    LinSysBlkRowProjection, LInSysBlkColProjection, LinSysPreconKrylov
