############################################################################################
## This file is part of RLinearAlgebra.jl
##
## Overview: abstractions and methods for stopping a randomized linear system solver
##
## Contents
## - Abstract Types
## - `check_stop_criterion` Docstring
## - Stopping Criteria
## - Export Statements
##
############################################################################################

# Dependencies:

#############################################
# Abstract Types
#############################################

"""
    LinSysStopCriterion

Abstract supertype for specifying stopping criteria for randomized linear solver.
"""
abstract type LinSysStopCriterion end

#############################################
# `check_stop_criterion` Docstring
#############################################
"""
    check_stop_criterion(
        log::LinSysSolverLog,
        stop::LinSysStopCriterion
        )

A common interface for specifying different strategies for stopping `log` with supertype
    `LinSysSolverLog`. A `stop` of supertype `LinSysStopCriterion` can be used to provide
    specific multiple implementations of stopping conditions for the same log type.
"""
function check_stop_criterion(
    log::LinSysSolverLog,
    stop::LinSysStopCriterion
)
    return nothing
end

#############################################
# Stopping Criteria
#############################################
include("linear_solver_stops/stop_max_iter.jl")
include("linear_solver_stops/stop_ma_thres.jl")
include("linear_solver_stops/stop_thres.jl")

#############################################
# Export Statements
#############################################
# See RLinearAlgebra.jl
#export LinSysStopCriterion
#export LSStopMaxIterations
