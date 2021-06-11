############################################################################################
## This file is part of RLinearAlgebra.jl
##
## Overview: abstractions and methods for routines used in solving linear systems of
## equations.
##
## Contents
## - Abstract Types
## - Vector Row Projection Solvers
## - Vector Column Projection Solvers
## - Block Row Projection Solvers
## - Block Column Projection Solvers
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

Abstract supertype for block row action projection methods.
"""
abstract type LinSysBlkColProjection <: LinSysSolveRoutine end

"""
    LinSysPreconKrylov <: LinSysSolveRoutine

Abstract supertype for block column action projection methods.
"""
abstract type LinSysPreconKrylov <: LinSysSolveRoutine end
