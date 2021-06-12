############################################################################################
## This file is part of RLinearAlgebra.jl
##
## Overview: abstractions and methods for stopping a randomized linear system solver
##
## Contents
## - Abstract Types
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
# Stopping Criteria
#############################################
include("linear_solver_stops/stop_max_iter.jl")

#############################################
# Export Statements
#############################################
# See RLinearAlgebra.jl
#export LinSysStopCriterion
#export LSStopMaxIterations
